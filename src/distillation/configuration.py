"""Configuration parsing, unit conversion, and engineering-stage validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray

from .components import ComponentSet, PropertyDataError, load_component_database


class ConfigurationError(ValueError):
    """Raised for invalid or under-specified column inputs."""


@dataclass(frozen=True, slots=True)
class Feed:
    """One feed on a one-based engineering stage number."""

    stage: int
    temperature_k: float
    pressure_pa: float
    liquid_component_flow_kmol_h: NDArray[np.float64]
    vapor_component_flow_kmol_h: NDArray[np.float64]
    name: str = "feed"

    @property
    def liquid_flow_kmol_h(self) -> float:
        return float(self.liquid_component_flow_kmol_h.sum())

    @property
    def vapor_flow_kmol_h(self) -> float:
        return float(self.vapor_component_flow_kmol_h.sum())

    @property
    def total_flow_kmol_h(self) -> float:
        return self.liquid_flow_kmol_h + self.vapor_flow_kmol_h

    @property
    def component_flow_kmol_h(self) -> NDArray[np.float64]:
        return self.liquid_component_flow_kmol_h + self.vapor_component_flow_kmol_h


@dataclass(frozen=True, slots=True)
class SolverSettings:
    max_iterations: int = 100
    temperature_tau_tolerance_k2: float = 1e-2
    vapor_absolute_tolerance: float = 1e-10
    bubble_point_tolerance: float = 1e-8
    max_inner_iterations: int = 200
    temperature_min_k: float = 120.0
    temperature_max_k: float = 800.0


@dataclass(frozen=True, slots=True)
class OperatingSpecification:
    kind: str
    value_kmol_h: float | None = None
    light_components: tuple[str | int, ...] = ()


@dataclass(frozen=True, slots=True)
class InitialGuess:
    kind: str = "auto"
    light_components: tuple[str | int, ...] = ()
    trace_component_flow_kmol_h: float = 1e-5
    top_temperature_k: float | None = None
    bottom_temperature_k: float | None = None


@dataclass(frozen=True, slots=True)
class ColumnConfig:
    name: str
    components: ComponentSet
    n_equilibrium_trays: int
    condenser_type: str
    reboiler_type: str
    stage_pressures_pa: NDArray[np.float64]
    feeds: tuple[Feed, ...]
    reflux_ratios: tuple[float, ...]
    operating_specification: OperatingSpecification
    solver: SolverSettings = field(default_factory=SolverSettings)
    initial_guess: InitialGuess = field(default_factory=InitialGuess)

    @property
    def n_stages(self) -> int:
        return self.n_equilibrium_trays + 2

    def with_reflux_ratios(self, values: Sequence[float]) -> "ColumnConfig":
        return replace(self, reflux_ratios=tuple(float(v) for v in values))


def _pressure_pa(data: Mapping[str, Any], prefix: str = "pressure") -> float:
    for suffix, factor in (("_pa", 1.0), ("_kpa", 1e3), ("_mpa", 1e6)):
        key = prefix + suffix
        if key in data:
            return float(data[key]) * factor
    raise ConfigurationError(f"Missing {prefix}_pa, {prefix}_kpa, or {prefix}_mpa")


def _build_pressures(data: Mapping[str, Any], n_stages: int) -> NDArray[np.float64]:
    if "stage_pressures_pa" in data or "stage_pressures_kpa" in data:
        key = "stage_pressures_pa" if "stage_pressures_pa" in data else "stage_pressures_kpa"
        factor = 1.0 if key.endswith("_pa") else 1e3
        result = np.asarray(data[key], dtype=float) * factor
    elif "top_kpa" in data and "bottom_kpa" in data:
        result = np.linspace(float(data["top_kpa"]), float(data["bottom_kpa"]), n_stages) * 1e3
    elif "anchors_kpa" in data:
        anchors = sorted((int(k), float(v)) for k, v in data["anchors_kpa"].items())
        if not anchors:
            raise ConfigurationError("pressure_profile.anchors_kpa must not be empty")
        stages = np.arange(1, n_stages + 1)
        ap = np.array([a[0] for a in anchors])
        av = np.array([a[1] for a in anchors])
        if ap.min() < 1 or ap.max() > n_stages:
            raise ConfigurationError("Pressure anchor stage is outside the modeled column")
        result = np.interp(stages, ap, av)
        drop = float(data.get("extrapolation_kpa_per_stage", 0.0))
        result[stages < ap[0]] = av[0] - drop * (ap[0] - stages[stages < ap[0]])
        result[stages > ap[-1]] = av[-1] + drop * (stages[stages > ap[-1]] - ap[-1])
        if bool(data.get("isobaric_end_stages", False)) and n_stages >= 4:
            result[0] = result[1]
            result[-1] = result[-2]
        result *= 1e3
    else:
        raise ConfigurationError("Unsupported pressure_profile; use explicit values, top/bottom, or anchors")
    if result.shape != (n_stages,):
        raise ConfigurationError(f"Expected {n_stages} stage pressures, got {result.size}")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ConfigurationError("Stage pressures must be finite and positive")
    return result


def _component_vector(
    value: Any,
    component_names: Sequence[str],
    field_name: str,
) -> NDArray[np.float64]:
    """Return a component vector from a positional list or name/value mapping."""

    n_components = len(component_names)
    if isinstance(value, Mapping):
        unknown = set(value) - set(component_names)
        if unknown:
            raise ConfigurationError(
                f"{field_name} contains unknown component(s) {sorted(unknown)}; "
                f"selected components are {list(component_names)}"
            )
        return np.array([float(value.get(name, 0.0)) for name in component_names])
    result = np.asarray(value, dtype=float)
    if result.shape != (n_components,):
        raise ConfigurationError(
            f"{field_name} must contain {n_components} values in component order "
            f"{list(component_names)}, or use a name/value mapping"
        )
    return result


def _parse_feed(data: Mapping[str, Any], component_names: Sequence[str], n_stages: int) -> Feed:
    n_components = len(component_names)
    stage = int(data["stage"])
    if stage < 1 or stage > n_stages:
        raise ConfigurationError(f"Feed stage {stage} is outside engineering stages 1..{n_stages}")
    if "liquid_component_flow_kmol_h" in data or "vapor_component_flow_kmol_h" in data:
        liquid = _component_vector(
            data.get("liquid_component_flow_kmol_h", {}),
            component_names,
            "liquid_component_flow_kmol_h",
        )
        vapor = _component_vector(
            data.get("vapor_component_flow_kmol_h", {}),
            component_names,
            "vapor_component_flow_kmol_h",
        )
    else:
        total = float(data["total_flow_kmol_h"])
        composition = _component_vector(data["composition"], component_names, "composition")
        if not np.isclose(composition.sum(), 1.0, atol=1e-10):
            raise ConfigurationError("Overall feed composition must have one normalized value per component")
        vapor_fraction = float(data.get("vapor_fraction", 0.0))
        if vapor_fraction < 0 or vapor_fraction > 1:
            raise ConfigurationError("Feed vapor_fraction must be between zero and one")
        vapor = total * vapor_fraction * composition
        liquid = total * (1.0 - vapor_fraction) * composition
    if liquid.shape != (n_components,) or vapor.shape != (n_components,):
        raise ConfigurationError(f"Feed vectors must each contain {n_components} component flows")
    if np.any(liquid < 0) or np.any(vapor < 0) or not np.all(np.isfinite(liquid + vapor)):
        raise ConfigurationError("Feed component flows must be finite and nonnegative")
    if liquid.sum() + vapor.sum() <= 0:
        raise ConfigurationError("Feed total flow must be positive")
    return Feed(
        name=str(data.get("name", f"feed_stage_{stage}")),
        stage=stage,
        temperature_k=float(data["temperature_k"]),
        pressure_pa=_pressure_pa(data),
        liquid_component_flow_kmol_h=liquid,
        vapor_component_flow_kmol_h=vapor,
    )


def config_from_mapping(data: Mapping[str, Any]) -> ColumnConfig:
    component_data: dict[str, Any] = {"components": data["components"]}
    if "binary_interaction_parameters" in data:
        component_data["binary_interaction_parameters"] = data["binary_interaction_parameters"]
    components = ComponentSet.from_mapping(component_data)
    n_trays = int(data["column"]["equilibrium_trays"])
    if n_trays < 1:
        raise ConfigurationError("column.equilibrium_trays must be positive")
    n_stages = n_trays + 2
    pressures = _build_pressures(data["column"]["pressure_profile"], n_stages)
    feeds = tuple(_parse_feed(f, components.names, n_stages) for f in data.get("feeds", []))
    if not feeds:
        raise ConfigurationError("At least one feed is required")
    spec_data = data["operating_specification"]
    spec = OperatingSpecification(
        kind=str(spec_data["kind"]),
        value_kmol_h=(float(spec_data["value_kmol_h"]) if "value_kmol_h" in spec_data else None),
        light_components=tuple(spec_data.get("light_components", ())),
    )
    initial_data = data.get("initial_guess", {})
    initial = InitialGuess(
        kind=str(initial_data.get("kind", "auto")),
        light_components=tuple(initial_data.get("light_components", spec.light_components)),
        trace_component_flow_kmol_h=float(initial_data.get("trace_component_flow_kmol_h", 1e-5)),
        top_temperature_k=(
            float(initial_data["top_temperature_k"]) if "top_temperature_k" in initial_data else None
        ),
        bottom_temperature_k=(
            float(initial_data["bottom_temperature_k"]) if "bottom_temperature_k" in initial_data else None
        ),
    )
    solver_data = data.get("solver", {})
    allowed = SolverSettings.__dataclass_fields__.keys()
    unknown = set(solver_data) - set(allowed)
    if unknown:
        raise ConfigurationError(f"Unknown solver setting(s): {sorted(unknown)}")
    solver = SolverSettings(**solver_data)
    reflux = data.get("reflux_ratios", [data.get("reflux_ratio")])
    if any(v is None for v in reflux) or any(float(v) <= 0 for v in reflux):
        raise ConfigurationError("Every reflux ratio must be positive")
    condenser = str(data["column"].get("condenser_type", "total"))
    reboiler = str(data["column"].get("reboiler_type", "partial"))
    if condenser != "total" or reboiler != "partial":
        raise ConfigurationError("This solver currently supports a total condenser and partial reboiler")
    return ColumnConfig(
        name=str(data.get("name", "distillation_column")),
        components=components,
        n_equilibrium_trays=n_trays,
        condenser_type=condenser,
        reboiler_type=reboiler,
        stage_pressures_pa=pressures,
        feeds=feeds,
        reflux_ratios=tuple(float(v) for v in reflux),
        operating_specification=spec,
        solver=solver,
        initial_guess=initial,
    )


def load_config(path: str | Path) -> ColumnConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        if path.suffix.lower() == ".json":
            data = json.load(stream)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(stream)
        else:
            raise ConfigurationError("Configuration file must be YAML or JSON")
    if not isinstance(data, Mapping):
        raise ConfigurationError("Configuration root must be a mapping")
    resolved = dict(data)
    component_entries = resolved.get("components")
    if isinstance(component_entries, list) and component_entries and all(
        isinstance(item, str) for item in component_entries
    ):
        database_reference = resolved.get("component_database")
        if not database_reference:
            raise ConfigurationError(
                "A component_database path is required when components are selected by name"
            )
        database_path = Path(str(database_reference))
        if not database_path.is_absolute():
            database_path = path.parent / database_path
        try:
            database = load_component_database(database_path)
        except (OSError, PropertyDataError) as exc:
            raise ConfigurationError(f"Could not load component database {database_path}: {exc}") from exc
        missing = [name for name in component_entries if name not in database]
        if missing:
            raise ConfigurationError(
                f"Component(s) not found in {database_path}: {missing}; available: {sorted(database)}"
            )
        resolved["components"] = [database[name] for name in component_entries]
    return config_from_mapping(resolved)
