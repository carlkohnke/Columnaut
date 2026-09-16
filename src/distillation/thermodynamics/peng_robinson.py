"""Peng-Robinson equation of state and departure-property calculations."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from ..components import ComponentSet
from .base import EOSState, ThermodynamicError, ThermodynamicPackage


class PengRobinson(ThermodynamicPackage):
    """Classical quadratic-mixing Peng-Robinson package.

    Binary interaction parameters are supported through the classical
    quadratic mixing rule. Liquid and vapor states use the smallest and largest
    physical real roots, respectively.
    """

    R_J_MOL_K = 8.314
    REFERENCE_TEMPERATURE_K = 298.15

    def __init__(self, components: ComponentSet) -> None:
        self.components = components
        self._tc = components.tc.copy()
        self._m = 0.37464 + 1.54226 * components.omega - 0.26992 * components.omega**2
        self._bi = 0.07780 * self.R_J_MOL_K * self._tc / components.pc

    @lru_cache(maxsize=4096)
    def _pure_parameters(self, temperature_k: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t = float(temperature_k)
        alpha = (1.0 + self._m * (1.0 - np.sqrt(t / self._tc))) ** 2
        ai = (
            0.45724
            * self.R_J_MOL_K**2
            * self._tc**2
            / self.components.pc
            * alpha
        )
        return alpha, ai

    def state(
        self,
        pressure_pa: float,
        temperature_k: float,
        composition: NDArray[np.float64],
        phase: str,
        *,
        _with_enthalpy: bool = True,
    ) -> EOSState:
        z = np.asarray(composition, dtype=float)
        n = len(self.components)
        if z.shape != (n,):
            raise ThermodynamicError(f"Composition shape {z.shape} does not match {n} components")
        if not np.all(np.isfinite(z)) or np.any(z < 0):
            raise ThermodynamicError(f"Non-finite or negative {phase}-phase composition: {z}")
        if pressure_pa <= 0 or temperature_k <= 0:
            raise ThermodynamicError(f"Pressure and temperature must be positive, got {pressure_pa}, {temperature_k}")
        alpha, ai = self._pure_parameters(float(temperature_k))
        q = np.sqrt(np.outer(ai, ai)) * (1.0 - self.components.kij)
        a = float(z @ q @ z)
        b = float(z @ self._bi)
        if b <= 0 or a <= 0:
            raise ThermodynamicError(f"Invalid Peng-Robinson mixture parameters a={a}, b={b}")

        rt_over_p = self.R_J_MOL_K * temperature_k / pressure_pa
        coefficients = np.array(
            [
                1.0,
                b - rt_over_p,
                -3.0 * b**2 - 2.0 * rt_over_p * b + a / pressure_pa,
                b**3 + rt_over_p * b**2 - a * b / pressure_pa,
            ]
        )
        all_roots = np.roots(coefficients)
        real_roots = np.real(all_roots[np.abs(np.imag(all_roots)) < 1e-8])
        if real_roots.size == 0:
            raise ThermodynamicError(
                f"No real EOS volume root at P={pressure_pa:g} Pa, T={temperature_k:g} K; roots={all_roots}"
            )
        real_roots.sort()
        eligible = real_roots[real_roots > b * (1.0 + 1e-12)]
        if eligible.size == 0:
            raise ThermodynamicError(
                f"No physical EOS root greater than b={b:g} at P={pressure_pa:g}, T={temperature_k:g}"
            )
        phase_u = phase.upper()
        if phase_u == "L":
            volume = float(eligible[0])
        elif phase_u == "V":
            volume = float(eligible[-1])
        else:
            raise ThermodynamicError("phase must be 'L' or 'V'")
        compressibility = pressure_pa * volume / (self.R_J_MOL_K * temperature_k)

        e = 1.0 - np.sqrt(2.0)
        s = 1.0 + np.sqrt(2.0)
        abar = 2.0 * (z @ q) - a
        try:
            log_repulsive = np.log((volume - b) * compressibility / volume)
            log_attractive = np.log((volume + s * b) / (volume + e * b))
            exponent = (
                (compressibility - 1.0) * self._bi / b
                - log_repulsive
                + (a / (b * self.R_J_MOL_K * temperature_k))
                / (e - s)
                * log_attractive
                * (1.0 + abar / a - self._bi / b)
            )
            phi = np.exp(exponent)
        except FloatingPointError as exc:
            raise ThermodynamicError(
                f"Fugacity calculation failed for phase={phase_u}, P={pressure_pa:g}, T={temperature_k:g}"
            ) from exc
        if not np.all(np.isfinite(phi)) or np.any(phi <= 0):
            raise ThermodynamicError(
                f"Invalid fugacity coefficients for phase={phase_u}, P={pressure_pa:g}, T={temperature_k:g}: {phi}"
            )

        enthalpy = np.nan
        if _with_enthalpy:
            tc = self._tc
            pc = self.components.pc
            root_alpha = np.sqrt(alpha)
            dqd_t = (
                0.45724
                * self.R_J_MOL_K**2
                * (self.components.kij - 1.0)
                * np.outer(tc, tc)
                / np.sqrt(np.outer(pc, pc))
                * (1.0 / (2.0 * np.sqrt(temperature_k)))
                * (
                    np.outer(self._m / np.sqrt(tc), root_alpha)
                    + np.outer(root_alpha, self._m / np.sqrt(tc))
                )
            )
            departure_derivative_term = float(z @ (q - temperature_k * dqd_t) @ z)
            residual_h = (
                self.R_J_MOL_K * temperature_k * (compressibility - 1.0)
                - departure_derivative_term
                / (2.0 * (s - 1.0) * b)
                * log_attractive
            )
            cp = self.components.cp
            tr = self.REFERENCE_TEMPERATURE_K
            ideal_terms = (
                self.components.href
                + cp[:, 0] * (temperature_k - tr)
                + cp[:, 1]
                * cp[:, 2]
                * (1.0 / np.tanh(cp[:, 2] / temperature_k) - 1.0 / np.tanh(cp[:, 2] / tr))
                + cp[:, 3]
                * cp[:, 4]
                * (1.0 / np.tanh(cp[:, 4] / temperature_k) - 1.0 / np.tanh(cp[:, 4] / tr))
            )
            enthalpy = float(z @ ideal_terms + residual_h)
        return EOSState(volume, float(compressibility), phi, enthalpy, real_roots)

    def fugacity_coefficients(
        self, pressure_pa: float, temperature_k: float, composition: NDArray[np.float64], phase: str
    ) -> NDArray[np.float64]:
        return self.state(
            pressure_pa, temperature_k, composition, phase, _with_enthalpy=False
        ).fugacity_coefficients

    def k_values(
        self,
        pressure_pa: float,
        temperature_k: float,
        liquid_composition: NDArray[np.float64],
        vapor_composition: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        phi_l = self.fugacity_coefficients(pressure_pa, temperature_k, liquid_composition, "L")
        phi_v = self.fugacity_coefficients(pressure_pa, temperature_k, vapor_composition, "V")
        return phi_l / phi_v

    def enthalpy(
        self, pressure_pa: float, temperature_k: float, composition: NDArray[np.float64], phase: str
    ) -> float:
        return self.state(pressure_pa, temperature_k, composition, phase).enthalpy_j_mol
