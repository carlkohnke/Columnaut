from __future__ import annotations

import numpy as np
import pytest

from distillation.components import Component, ComponentSet
from distillation.thermodynamics.equilibrium import bracketed_temperature_root
from distillation.thermodynamics.peng_robinson import PengRobinson


def test_pure_component_liquid_and_vapor_roots():
    propane = Component(
        "propane", 369.83, 4.248e6, 0.1523, (51.92, 192.45, 1.6265, 116.8, 0.7236), -104700
    )
    package = PengRobinson(ComponentSet([propane]))
    liquid = package.state(1.0e5, 250.0, np.array([1.0]), "L")
    vapor = package.state(1.0e5, 250.0, np.array([1.0]), "V")
    assert liquid.roots_m3_mol.size == 3
    assert liquid.molar_volume_m3_mol < vapor.molar_volume_m3_mol
    assert liquid.compressibility < vapor.compressibility
    assert np.all(liquid.fugacity_coefficients > 0)


def test_mixture_properties_regression(hydrocarbon_config):
    package = PengRobinson(hydrocarbon_config.components)
    x = np.array([1.00, 2.34, 2.43, 4.46, 2.01, 1.08]) / 13.32
    y = np.array([1.21, 0.84, 1.22, 0.50, 0.29, 0.04]) / 4.10
    expected_k = np.array([3.93960449, 1.16368103, 1.63445003, 0.36043376, 0.47268806, 0.11507507])
    assert np.allclose(package.k_values(225000.0, 301.0, x, y), expected_k, rtol=2e-8, atol=2e-8)
    assert package.enthalpy(225000.0, 301.0, x, "L") == pytest.approx(-162580.874733443, abs=2e-8)
    assert package.enthalpy(225000.0, 301.0, y, "V") == pytest.approx(-126049.879878240, abs=2e-8)


def test_property_precision_is_retained(hydrocarbon_config):
    package = PengRobinson(hydrocarbon_config.components)
    assert package._tc[0] == pytest.approx(369.83)


def test_bubble_and_dew_residual_roots(hydrocarbon_config):
    package = PengRobinson(hydrocarbon_config.components)
    z = np.array([0.24447, 0.35177, 0.40376, 1e-7, 1e-7, 1e-7])
    z /= z.sum()
    dew = bracketed_temperature_root(
        lambda t: package.dew_point_residual(216726.288, t, z, z), 200, 400, initial_k=280
    )
    assert abs(package.dew_point_residual(216726.288, dew, z, z)) < 1e-8
    heavy = np.array([1e-7, 1e-7, 1e-7, 0.59224, 0.27463, 0.13313])
    heavy /= heavy.sum()
    bubble = bracketed_temperature_root(
        lambda t: package.bubble_point_residual(236652.664, t, heavy, heavy),
        250,
        450,
        initial_k=330,
    )
    assert abs(package.bubble_point_residual(236652.664, bubble, heavy, heavy)) < 1e-8
