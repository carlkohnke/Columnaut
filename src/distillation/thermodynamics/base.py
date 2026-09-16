"""Thermodynamic-package interface used by the column solver."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class ThermodynamicError(RuntimeError):
    """Property calculation failed with phase/state context."""


@dataclass(frozen=True, slots=True)
class EOSState:
    molar_volume_m3_mol: float
    compressibility: float
    fugacity_coefficients: NDArray[np.float64]
    enthalpy_j_mol: float
    roots_m3_mol: NDArray[np.float64]


class ThermodynamicPackage(ABC):
    """Minimal interface required by the distillation solver."""

    @abstractmethod
    def fugacity_coefficients(
        self, pressure_pa: float, temperature_k: float, composition: NDArray[np.float64], phase: str
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def k_values(
        self,
        pressure_pa: float,
        temperature_k: float,
        liquid_composition: NDArray[np.float64],
        vapor_composition: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def enthalpy(
        self, pressure_pa: float, temperature_k: float, composition: NDArray[np.float64], phase: str
    ) -> float: ...

    def bubble_point_residual(
        self,
        pressure_pa: float,
        temperature_k: float,
        liquid_composition: NDArray[np.float64],
        vapor_composition: NDArray[np.float64],
    ) -> float:
        return float(
            np.dot(
                self.k_values(pressure_pa, temperature_k, liquid_composition, vapor_composition),
                liquid_composition,
            )
            - 1.0
        )

    def dew_point_residual(
        self,
        pressure_pa: float,
        temperature_k: float,
        liquid_composition: NDArray[np.float64],
        vapor_composition: NDArray[np.float64],
    ) -> float:
        k = self.k_values(pressure_pa, temperature_k, liquid_composition, vapor_composition)
        return float(np.sum(vapor_composition / k) - 1.0)
