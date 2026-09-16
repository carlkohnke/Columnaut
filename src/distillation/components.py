"""Component property storage and validation.

The package uses SI internally: Pa, K, J/mol, and kmol/h.  J/mol and
kJ/kmol have the same numerical value, but conversion helpers elsewhere make
that identity explicit at energy-rate boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray


class PropertyDataError(ValueError):
    """Raised when component property data are incomplete or nonphysical."""


@dataclass(frozen=True, slots=True)
class Component:
    name: str
    critical_temperature_k: float
    critical_pressure_pa: float
    acentric_factor: float
    ideal_gas_cp_coefficients: tuple[float, float, float, float, float]
    reference_enthalpy_j_mol: float
    identifier: str | int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise PropertyDataError("Component name must not be empty")
        if not np.isfinite(self.critical_temperature_k) or self.critical_temperature_k <= 0:
            raise PropertyDataError(f"{self.name}: critical_temperature_k must be positive")
        if not np.isfinite(self.critical_pressure_pa) or self.critical_pressure_pa <= 0:
            raise PropertyDataError(f"{self.name}: critical_pressure_pa must be positive")
        if not np.isfinite(self.acentric_factor):
            raise PropertyDataError(f"{self.name}: acentric_factor must be finite")
        if len(self.ideal_gas_cp_coefficients) != 5 or not np.all(
            np.isfinite(self.ideal_gas_cp_coefficients)
        ):
            raise PropertyDataError(f"{self.name}: exactly five finite heat-capacity coefficients are required")
        if not np.isfinite(self.reference_enthalpy_j_mol):
            raise PropertyDataError(f"{self.name}: reference_enthalpy_j_mol must be finite")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Component":
        pressure_keys = [
            key
            for key in ("critical_pressure_pa", "critical_pressure_kpa", "critical_pressure_mpa")
            if key in data
        ]
        if len(pressure_keys) != 1:
            raise PropertyDataError(
                f"{data.get('name', '<unnamed>')}: provide exactly one critical-pressure unit field"
            )
        key = pressure_keys[0]
        scale = {"critical_pressure_pa": 1.0, "critical_pressure_kpa": 1e3, "critical_pressure_mpa": 1e6}[key]
        cp = tuple(float(v) for v in data["ideal_gas_cp_coefficients"])
        if len(cp) != 5:
            raise PropertyDataError("ideal_gas_cp_coefficients must contain five values")
        href = data.get("reference_enthalpy_j_mol", data.get("enthalpy_of_formation_j_mol"))
        if href is None:
            raise PropertyDataError(
                f"{data.get('name', '<unnamed>')}: reference_enthalpy_j_mol is required"
            )
        return cls(
            name=str(data["name"]),
            identifier=data.get("identifier"),
            critical_temperature_k=float(data["critical_temperature_k"]),
            critical_pressure_pa=float(data[key]) * scale,
            acentric_factor=float(data["acentric_factor"]),
            ideal_gas_cp_coefficients=cp,  # type: ignore[arg-type]
            reference_enthalpy_j_mol=float(href),
        )


class ComponentSet:
    """Validated component collection with dense property arrays."""

    def __init__(
        self,
        components: Sequence[Component],
        binary_interaction: Iterable[Iterable[float]] | None = None,
    ) -> None:
        if not components:
            raise PropertyDataError("At least one component is required")
        self.components = tuple(components)
        names = [c.name for c in components]
        if len(set(names)) != len(names):
            raise PropertyDataError("Component names must be unique")
        self.names = tuple(names)
        self.tc: NDArray[np.float64] = np.array([c.critical_temperature_k for c in components])
        self.pc: NDArray[np.float64] = np.array([c.critical_pressure_pa for c in components])
        self.omega: NDArray[np.float64] = np.array([c.acentric_factor for c in components])
        self.cp: NDArray[np.float64] = np.array([c.ideal_gas_cp_coefficients for c in components])
        self.href: NDArray[np.float64] = np.array([c.reference_enthalpy_j_mol for c in components])
        n = len(components)
        if binary_interaction is None:
            kij = np.zeros((n, n), dtype=float)
        else:
            kij = np.asarray(binary_interaction, dtype=float)
            if kij.shape != (n, n):
                raise PropertyDataError(f"binary_interaction must have shape {(n, n)}, got {kij.shape}")
            if not np.all(np.isfinite(kij)) or not np.allclose(kij, kij.T, atol=1e-14):
                raise PropertyDataError("binary_interaction must be finite and symmetric")
            if not np.allclose(np.diag(kij), 0.0, atol=1e-14):
                raise PropertyDataError("binary_interaction diagonal must be zero")
        self.kij: NDArray[np.float64] = kij

    def __len__(self) -> int:
        return len(self.components)

    def indices(self, names_or_indices: Sequence[str | int]) -> tuple[int, ...]:
        result: list[int] = []
        for value in names_or_indices:
            if isinstance(value, str):
                try:
                    result.append(self.names.index(value))
                except ValueError as exc:
                    raise PropertyDataError(f"Unknown component {value!r}; available: {self.names}") from exc
            else:
                index = int(value)
                if index < 0 or index >= len(self):
                    raise PropertyDataError(f"Component index {index} is out of range")
                result.append(index)
        return tuple(result)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentSet":
        components = [
            item if isinstance(item, Component) else Component.from_mapping(item)
            for item in data["components"]
        ]
        return cls(components, data.get("binary_interaction_parameters"))


def load_component_database(path: str | Path) -> dict[str, Component]:
    """Load a YAML/JSON property database and index components by name.

    The file root may be a list of component mappings or a mapping containing
    a ``components`` list. Each entry uses the same fields accepted by
    :meth:`Component.from_mapping`.
    """

    database_path = Path(path)
    with database_path.open("r", encoding="utf-8") as stream:
        if database_path.suffix.lower() == ".json":
            data = json.load(stream)
        elif database_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(stream)
        else:
            raise PropertyDataError("Component database must be YAML or JSON")
    entries = data.get("components") if isinstance(data, Mapping) else data
    if not isinstance(entries, list) or not entries:
        raise PropertyDataError("Component database must contain a non-empty components list")
    components = [Component.from_mapping(entry) for entry in entries]
    result = {component.name: component for component in components}
    if len(result) != len(components):
        raise PropertyDataError("Component database names must be unique")
    return result
