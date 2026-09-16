"""Robust sequential bubble-point column solver."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .balances import BalanceSolveError, solve_lower_bidiagonal, solve_tridiagonal, tridiagonal_residual
from .column import ColumnModel
from .configuration import ColumnConfig
from .results import SimulationBatch, SimulationResult
from .thermodynamics.base import ThermodynamicError, ThermodynamicPackage
from .thermodynamics.equilibrium import bracketed_temperature_root
from .thermodynamics.peng_robinson import PengRobinson


class ConvergenceError(RuntimeError):
    """The column did not converge within a configured finite iteration limit."""


def j_per_mol_times_kmol_per_h_to_kj_per_h(
    molar_enthalpy_j_mol: float | NDArray[np.float64], molar_flow_kmol_h: float | NDArray[np.float64]
) -> float | NDArray[np.float64]:
    """Convert h[J/mol] * flow[kmol/h] to kJ/h.

    The factors 1000 mol/kmol and 1 kJ/1000 J cancel numerically.  Keeping
    this named boundary avoids relying on that coincidence silently.
    """

    return np.asarray(molar_enthalpy_j_mol) * np.asarray(molar_flow_kmol_h)


@dataclass(slots=True)
class _State:
    temperature: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    k_values: NDArray[np.float64]
    liquid_flow: NDArray[np.float64]
    vapor_flow: NDArray[np.float64]
    h_liquid: NDArray[np.float64]
    h_vapor: NDArray[np.float64]
    h_feed: NDArray[np.float64]
    q_stage: NDArray[np.float64]


class ColumnSolver:
    """Solver that depends only on :class:`ThermodynamicPackage`."""

    def __init__(self, config: ColumnConfig, thermodynamics: ThermodynamicPackage | None = None) -> None:
        self.config = config
        self.model = ColumnModel.from_config(config)
        self.thermo = thermodynamics or PengRobinson(config.components)
        self.ns = config.n_stages
        self.nc = len(config.components)
        self.pressure = np.asarray(config.stage_pressures_pa, dtype=float)

    def _split_initial_products(
        self, distillate: float, bottoms: float, light: tuple[int, ...]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, ...]]:
        if not light:
            requested = self.config.initial_guess.light_components
            if requested:
                light = self.config.components.indices(requested)
            else:
                # Generic automatic initializer: the lower-boiling half by Tc.
                order = np.argsort(self.config.components.tc)
                light = tuple(int(i) for i in order[: max(1, self.nc // 2)])
        heavy = tuple(i for i in range(self.nc) if i not in light)
        trace = self.config.initial_guess.trace_component_flow_kmol_h
        total = self.model.total_component_feed
        dvec = np.full(self.nc, trace, dtype=float)
        bvec = np.full(self.nc, trace, dtype=float)
        dvec[list(light)] = total[list(light)]
        bvec[list(heavy)] = total[list(heavy)]
        zd = dvec / dvec.sum()
        zb = bvec / bvec.sum()
        return zd, zb, light

    def _initial_temperature(
        self, zd: NDArray[np.float64], zb: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        settings = self.config.solver
        feed_temperatures = [feed.temperature_k for feed in self.config.feeds]
        top_guess = self.config.initial_guess.top_temperature_k
        bottom_guess = self.config.initial_guess.bottom_temperature_k
        if top_guess is None:
            top_guess = min(feed_temperatures) - 20.0
        if bottom_guess is None:
            bottom_guess = max(feed_temperatures) + 20.0
        top = bracketed_temperature_root(
            lambda t: self.thermo.dew_point_residual(self.pressure[0], t, zd, zd),
            settings.temperature_min_k,
            settings.temperature_max_k,
            initial_k=top_guess,
        )
        bottom = bracketed_temperature_root(
            lambda t: self.thermo.bubble_point_residual(self.pressure[-1], t, zb, zb),
            settings.temperature_min_k,
            settings.temperature_max_k,
            initial_k=bottom_guess,
        )
        anchors_stage = [0]
        anchors_temp = [top]
        for feed in sorted(self.config.feeds, key=lambda f: f.stage):
            anchors_stage.append(feed.stage - 1)
            anchors_temp.append(feed.temperature_k)
        anchors_stage.append(self.ns - 1)
        anchors_temp.append(bottom)
        # Duplicated-stage feeds were already checked and are collapsed here.
        unique: dict[int, float] = {}
        for stage, temp in zip(anchors_stage, anchors_temp, strict=True):
            unique[stage] = temp
        stages = np.arange(self.ns)
        keys = np.array(sorted(unique))
        vals = np.array([unique[k] for k in keys])
        return np.interp(stages, keys, vals)

    def _initial_compositions(
        self, zd: NDArray[np.float64], zb: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        anchor_stage = [0]
        anchor_x = [zd]
        anchor_y = [zd]
        feed_stages = np.flatnonzero(self.model.feed_total > 0)
        for s in feed_stages:
            z = self.model.overall_feed_composition[s]
            anchor_stage.append(int(s))
            anchor_x.append(self.model.liquid_feed_composition[s] if self.model.feed_liquid[s] > 0 else z)
            anchor_y.append(self.model.vapor_feed_composition[s] if self.model.feed_vapor[s] > 0 else z)
        anchor_stage.append(self.ns - 1)
        anchor_x.append(zb)
        anchor_y.append(zb)
        stages = np.arange(self.ns)
        x = np.empty((self.ns, self.nc))
        y = np.empty_like(x)
        a = np.asarray(anchor_stage)
        for i in range(self.nc):
            x[:, i] = np.interp(stages, a, np.array([v[i] for v in anchor_x]))
            y[:, i] = np.interp(stages, a, np.array([v[i] for v in anchor_y]))
        return x, y

    def _initial_vapor_flow(self, reflux_ratio: float, distillate: float) -> NDArray[np.float64]:
        v = np.zeros(self.ns + 1)
        v[1 : self.ns] = (reflux_ratio + 1.0) * distillate
        feeds = sorted(self.config.feeds, key=lambda f: f.stage)
        for feed in feeds:
            start = feed.stage - 1
            v[start : self.ns] -= feed.vapor_flow_kmol_h
        return v

    def _all_k(
        self, temperature: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        values = np.empty_like(x)
        for stage in range(self.ns):
            try:
                values[stage] = self.thermo.k_values(
                    self.pressure[stage], temperature[stage], x[stage], y[stage]
                )
            except ThermodynamicError as exc:
                raise ThermodynamicError(f"Stage {stage + 1} K-value calculation failed: {exc}") from exc
        return values

    def _find_vapor_equilibrium(
        self,
        temperature: NDArray[np.float64],
        x: NDArray[np.float64],
        y_start: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        y_out = np.empty_like(x)
        k_out = np.empty_like(x)
        counts = np.zeros(self.ns, dtype=np.int64)
        settings = self.config.solver
        for stage in range(self.ns):
            temporary = y_start[stage].copy()
            for inner in range(1, settings.max_inner_iterations + 1):
                try:
                    kval = self.thermo.k_values(
                        self.pressure[stage], temperature[stage], x[stage], temporary
                    )
                except ThermodynamicError as exc:
                    raise ThermodynamicError(
                        f"Stage {stage + 1} vapor iteration {inner} failed: {exc}"
                    ) from exc
                tester = x[stage] * kval
                total = float(tester.sum())
                if not np.isfinite(total) or total <= 0:
                    raise ConvergenceError(
                        f"Stage {stage + 1} vapor iteration produced invalid sum {total:g}"
                    )
                tester /= total
                metric = float(np.max(np.abs(tester - temporary)))
                converged = bool(metric <= settings.vapor_absolute_tolerance)
                if converged:
                    y_out[stage] = tester / tester.sum()
                    k_out[stage] = kval
                    counts[stage] = inner
                    break
                temporary = tester
            else:
                worst = int(np.argmax(np.abs(tester - temporary)))
                raise ConvergenceError(
                    f"Vapor composition failed on stage {stage + 1}, component "
                    f"{self.config.components.names[worst]!r}, after {settings.max_inner_iterations} iterations"
                )
        return y_out, k_out, counts

    def _component_balance(
        self, k_values: NDArray[np.float64], vapor_flow: NDArray[np.float64], side_draw: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[tuple[NDArray[np.float64], ...]]]:
        f = self.model.feed_total
        z = self.model.overall_feed_composition
        cum_f = np.cumsum(f)
        cum_u = np.cumsum(side_draw)
        x = np.empty((self.ns, self.nc))
        systems: list[tuple[NDArray[np.float64], ...]] = []
        for comp in range(self.nc):
            lower_full = np.zeros(self.ns)
            lower_full[1:] = vapor_flow[1 : self.ns] + cum_f[:-1] - cum_u[:-1]
            diagonal = -(
                vapor_flow[1 : self.ns + 1]
                + cum_f
                - cum_u
                + side_draw
                + vapor_flow[: self.ns] * k_values[:, comp]
            )
            upper_full = np.zeros(self.ns)
            upper_full[:-1] = vapor_flow[1 : self.ns] * k_values[1:, comp]
            rhs = -f * z[:, comp]
            try:
                x[:, comp] = solve_tridiagonal(
                    lower_full[1:], diagonal, upper_full[:-1], rhs
                )
            except BalanceSolveError as exc:
                raise BalanceSolveError(
                    f"Component balance failed for {self.config.components.names[comp]!r}: {exc}"
                ) from exc
            systems.append((lower_full[1:].copy(), diagonal.copy(), upper_full[:-1].copy(), rhs.copy()))
        totals = x.sum(axis=1)
        bad = np.flatnonzero(~np.isfinite(totals) | (np.abs(totals) < 1e-16))
        if bad.size:
            raise BalanceSolveError(f"Cannot normalize liquid composition on stage {bad[0] + 1}")
        x /= totals[:, None]
        minimum = float(x.min())
        if minimum < -1e-12:
            s, c = np.unravel_index(int(np.argmin(x)), x.shape)
            raise BalanceSolveError(
                f"Negative composition {x[s,c]:g} on stage {s+1}, "
                f"component {self.config.components.names[c]!r}"
            )
        x = np.maximum(x, 0.0)
        x /= x.sum(axis=1, keepdims=True)
        return x, systems

    def _new_temperature(
        self, previous: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        result = np.empty(self.ns)
        settings = self.config.solver
        for stage in range(self.ns):
            try:
                result[stage] = bracketed_temperature_root(
                    lambda t, s=stage: self.thermo.bubble_point_residual(
                        self.pressure[s], t, x[s], y[s]
                    ),
                    settings.temperature_min_k,
                    settings.temperature_max_k,
                    xtol=settings.bubble_point_tolerance * 0.1,
                    initial_k=float(previous[stage]),
                )
            except (ThermodynamicError, ConvergenceError) as exc:
                raise ConvergenceError(f"Temperature solve failed at stage {stage + 1}: {exc}") from exc
        return result

    def _enthalpies_and_flows(
        self,
        temperature: NDArray[np.float64],
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        vapor_flow: NDArray[np.float64],
        side_draw: NDArray[np.float64],
        reflux_ratio: float,
        distillate: float,
        bottoms: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        h_v = np.empty(self.ns)
        h_l = np.empty(self.ns)
        for stage in range(self.ns):
            h_v[stage] = self.thermo.enthalpy(
                self.pressure[stage], temperature[stage], y[stage], "V"
            )
            h_l[stage] = self.thermo.enthalpy(
                self.pressure[stage], temperature[stage], x[stage], "L"
            )
        h_f = np.zeros(self.ns)
        for stage in np.flatnonzero(self.model.feed_total > 0):
            feed_t = self.model.feed_temperature_k[stage]
            feed_p = self.model.feed_pressure_pa[stage]
            total = self.model.feed_total[stage]
            liquid_term = 0.0
            vapor_term = 0.0
            if self.model.feed_liquid[stage] > 0:
                liquid_term = float(
                    j_per_mol_times_kmol_per_h_to_kj_per_h(
                        self.thermo.enthalpy(
                            feed_p,
                            feed_t,
                            self.model.liquid_feed_composition[stage],
                            "L",
                        ),
                        self.model.feed_liquid[stage],
                    )
                )
            if self.model.feed_vapor[stage] > 0:
                vapor_term = float(
                    j_per_mol_times_kmol_per_h_to_kj_per_h(
                        self.thermo.enthalpy(
                            feed_p,
                            feed_t,
                            self.model.vapor_feed_composition[stage],
                            "V",
                        ),
                        self.model.feed_vapor[stage],
                    )
                )
            h_f[stage] = (liquid_term + vapor_term) / total

        q = np.zeros(self.ns)
        q[0] = float(
            j_per_mol_times_kmol_per_h_to_kj_per_h(h_v[1], vapor_flow[1])
            - j_per_mol_times_kmol_per_h_to_kj_per_h(h_l[0], (reflux_ratio + 1.0) * distillate)
        )
        first = float(
            np.sum(j_per_mol_times_kmol_per_h_to_kj_per_h(h_f, self.model.feed_total))
            - np.sum(j_per_mol_times_kmol_per_h_to_kj_per_h(h_l, side_draw))
        )
        second = float(-np.sum(q[:-1]) - j_per_mol_times_kmol_per_h_to_kj_per_h(h_l[-1], bottoms))
        q[-1] = first + second

        alpha = np.zeros(self.ns)
        beta = np.zeros(self.ns)
        gamma = np.zeros(self.ns)
        cum_f = np.cumsum(self.model.feed_total)
        cum_u = np.cumsum(side_draw)
        alpha[1:-1] = h_l[:-2] - h_v[1:-1]
        beta[1:-1] = h_v[2:] - h_l[1:-1]
        for stage in range(1, self.ns - 1):
            gamma[stage] = (
                (cum_f[stage - 1] - cum_u[stage - 1]) * (h_l[stage] - h_l[0])
                + self.model.feed_total[stage] * (h_l[stage] - h_f[stage])
            )
        rhs = gamma[1:-1].copy()
        rhs[0] -= alpha[1] * vapor_flow[1]
        vapor_new = vapor_flow.copy()
        vapor_new[2 : self.ns] = solve_lower_bidiagonal(alpha[2:-1], beta[1:-1], rhs)
        liquid = vapor_new[1 : self.ns + 1] + cum_f - cum_u
        return h_l, h_v, h_f, q, liquid, vapor_new

    def solve(self, reflux_ratio: float) -> SimulationResult:
        started = perf_counter()
        distillate, bottoms, light = self.model.product_flows()
        zd, zb, _ = self._split_initial_products(distillate, bottoms, light)
        temperature = self._initial_temperature(zd, zb)
        x, y = self._initial_compositions(zd, zb)
        y, k_values, _ = self._find_vapor_equilibrium(temperature, x, y)
        x[0] = y[1]
        vapor = self._initial_vapor_flow(reflux_ratio, distillate)
        liquid = np.zeros(self.ns)
        side_draw = np.zeros(self.ns)
        side_draw[0] = distillate
        tau_history: list[float] = []
        inner_history: list[int] = []
        final_systems: list[tuple[NDArray[np.float64], ...]] = []
        h_l = h_v = h_f = q = np.zeros(self.ns)
        tolerance = self.config.solver.temperature_tau_tolerance_k2
        converged = False
        for iteration in range(1, self.config.solver.max_iterations + 1):
            k_values = self._all_k(temperature, x, y)
            x, final_systems = self._component_balance(k_values, vapor, side_draw)
            y, _, counts_a = self._find_vapor_equilibrium(temperature, x, y)
            x[0] = y[1]
            new_temperature = self._new_temperature(temperature, x, y)
            y, k_values, counts_b = self._find_vapor_equilibrium(new_temperature, x, y)
            x[0] = y[1]
            h_l, h_v, h_f, q, liquid, vapor = self._enthalpies_and_flows(
                new_temperature,
                x,
                y,
                vapor,
                side_draw,
                reflux_ratio,
                distillate,
                bottoms,
            )
            tau = float(np.sum((new_temperature - temperature) ** 2))
            last_temperature_delta = np.abs(new_temperature - temperature)
            tau_history.append(tau)
            inner_history.append(int(max(counts_a.max(), counts_b.max())))
            temperature = new_temperature
            if tau <= tolerance:
                converged = True
                break
        if not converged:
            worst_stage = int(np.argmax(last_temperature_delta)) + 1
            raise ConvergenceError(
                f"Column failed to converge after {self.config.solver.max_iterations} iterations; "
                f"last tau={tau_history[-1]:.6g} K^2 (limit {tolerance:.6g}), "
                f"worst stage={worst_stage}, max |dT|={last_temperature_delta.max():.6g} K"
            )

        # Polish each converged stage so K, y, and the bracketed bubble
        # temperature refer to the same state. The outer-loop tolerance
        # controls column coupling; this separate test enforces the
        # thermodynamic residual independently.
        for _ in range(8):
            temperature = self._new_temperature(temperature, x, y)
            y, k_values, _ = self._find_vapor_equilibrium(temperature, x, y)
            x[0] = y[1]
            bubble = np.sum(k_values * x, axis=1) - 1.0
            if float(np.max(np.abs(bubble))) <= 1e-6:
                break
        else:
            worst = int(np.argmax(np.abs(bubble)))
            raise ConvergenceError(
                f"Equilibrium polish failed on stage {worst + 1}; "
                f"bubble residual={bubble[worst]:.6g}"
            )
        h_l, h_v, h_f, q, liquid, vapor = self._enthalpies_and_flows(
            temperature,
            x,
            y,
            vapor,
            side_draw,
            reflux_ratio,
            distillate,
            bottoms,
        )

        diagnostics = self._diagnostics(
            temperature,
            x,
            y,
            k_values,
            liquid,
            vapor,
            side_draw,
            h_l,
            h_f,
            q,
            final_systems,
            tau_history,
            inner_history,
            distillate,
            bottoms,
        )
        q_condenser = float(q[0])
        q_reboiler = float(q[-1])
        reported_condenser = q_condenser
        reported_reboiler = q_reboiler
        return SimulationResult(
            name=self.config.name,
            reflux_ratio=float(reflux_ratio),
            component_names=self.config.components.names,
            stage_numbers=np.arange(1, self.ns + 1, dtype=np.int64),
            pressure_pa=self.pressure.copy(),
            temperature_k=temperature.copy(),
            liquid_composition=x.copy(),
            vapor_composition=y.copy(),
            k_values=k_values.copy(),
            liquid_flow_kmol_h=liquid.copy(),
            vapor_flow_kmol_h=vapor[: self.ns].copy(),
            feed_flow_kmol_h=self.model.feed_total.copy(),
            distillate_flow_kmol_h=distillate,
            bottoms_flow_kmol_h=bottoms,
            condenser_duty_rate_kj_h=q_condenser,
            reboiler_duty_rate_kj_h=q_reboiler,
            reported_condenser_duty_kj_h=float(reported_condenser),
            reported_reboiler_duty_kj_h=float(reported_reboiler),
            iterations=iteration,
            converged=True,
            solve_time_seconds=perf_counter() - started,
            diagnostics=diagnostics,
        )

    def _diagnostics(
        self,
        temperature: NDArray[np.float64],
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        k_values: NDArray[np.float64],
        liquid: NDArray[np.float64],
        vapor: NDArray[np.float64],
        side_draw: NDArray[np.float64],
        h_l: NDArray[np.float64],
        h_f: NDArray[np.float64],
        q: NDArray[np.float64],
        systems: list[tuple[NDArray[np.float64], ...]],
        tau_history: list[float],
        inner_history: list[int],
        distillate: float,
        bottoms: float,
    ) -> dict[str, Any]:
        del systems  # Rebuild with final flows/K; the solve systems precede the final flow update.
        balance = np.empty((self.ns, self.nc))
        cum_f = np.cumsum(self.model.feed_total)
        cum_u = np.cumsum(side_draw)
        for comp in range(self.nc):
            lower = vapor[1 : self.ns] + cum_f[:-1] - cum_u[:-1]
            diagonal = -(
                vapor[1 : self.ns + 1]
                + cum_f
                - cum_u
                + side_draw
                + vapor[: self.ns] * k_values[:, comp]
            )
            upper = vapor[1 : self.ns] * k_values[1:, comp]
            rhs = -self.model.feed_total * self.model.overall_feed_composition[:, comp]
            balance[:, comp] = tridiagonal_residual(lower, diagonal, upper, x[:, comp], rhs)
        bubble = np.sum(k_values * x, axis=1) - 1.0
        vle = y - k_values * x
        feed_energy = float(
            np.sum(j_per_mol_times_kmol_per_h_to_kj_per_h(h_f, self.model.feed_total))
        )
        product_energy = float(
            j_per_mol_times_kmol_per_h_to_kj_per_h(h_l[0], distillate)
            + j_per_mol_times_kmol_per_h_to_kj_per_h(h_l[-1], bottoms)
        )
        energy_residual = feed_energy - product_energy - float(q[0]) - float(q[-1])
        return {
            "temperature_tau_history_k2": tau_history,
            "max_inner_vapor_iterations_by_outer_iteration": inner_history,
            "liquid_composition_sum_error": np.abs(x.sum(axis=1) - 1.0),
            "vapor_composition_sum_error": np.abs(y.sum(axis=1) - 1.0),
            "component_balance_residual_kmol_h": balance,
            "max_component_balance_residual_kmol_h": float(np.max(np.abs(balance))),
            "overall_material_balance_residual_kmol_h": float(
                self.model.total_feed - distillate - bottoms
            ),
            "vle_residual": vle,
            "max_vle_residual": float(np.max(np.abs(vle))),
            "bubble_point_residual": bubble,
            "max_bubble_point_residual": float(np.max(np.abs(bubble))),
            "overall_energy_balance_residual_kj_h": energy_residual,
            "flows_finite": bool(np.all(np.isfinite(liquid)) and np.all(np.isfinite(vapor))),
            "temperatures_finite": bool(np.all(np.isfinite(temperature))),
            "duty_dimensional_note": (
                "Q stage values are kJ/h because J/mol multiplied by kmol/h is numerically kJ/h."
            ),
        }


def simulate(config: ColumnConfig, thermodynamics: ThermodynamicPackage | None = None) -> SimulationBatch:
    started = perf_counter()
    solver = ColumnSolver(config, thermodynamics)
    results = [solver.solve(ratio) for ratio in config.reflux_ratios]
    return SimulationBatch(results, perf_counter() - started)
