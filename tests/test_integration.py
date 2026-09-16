from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from distillation.configuration import load_config
from distillation.solver import ColumnSolver, ConvergenceError, simulate


def test_hydrocarbon_duties_and_invariants(hydrocarbon_config):
    config = hydrocarbon_config.with_reflux_ratios([2.4])
    result = simulate(config).results[0]
    assert result.converged
    assert result.diagnostics["max_bubble_point_residual"] <= 1e-6
    assert np.max(np.abs(result.liquid_composition.sum(axis=1) - 1.0)) <= 1e-10
    assert np.max(np.abs(result.vapor_composition.sum(axis=1) - 1.0)) <= 1e-10
    assert abs(result.diagnostics["overall_material_balance_residual_kmol_h"]) <= 1e-12
    assert abs(result.diagnostics["overall_energy_balance_residual_kj_h"]) <= 1e-6
    assert result.reported_condenser_duty_kj_h == result.condenser_duty_rate_kj_h
    assert result.reported_reboiler_duty_kj_h == result.reboiler_duty_rate_kj_h
    assert abs(result.reported_condenser_duty_kj_h) < 0.1 * 91_115_985.01


def test_three_component_different_tray_count(project_root):
    config = load_config(project_root / "examples" / "three_component_column.yaml")
    result = simulate(config).results[0]
    assert result.liquid_composition.shape == (10, 3)
    assert result.diagnostics["max_bubble_point_residual"] <= 1e-6
    assert np.allclose(result.liquid_composition.sum(axis=1), 1.0, atol=1e-10)


def test_additional_tray_count_is_not_hardcoded(project_root):
    config = load_config(project_root / "examples" / "three_component_column.yaml")
    expanded = replace(
        config,
        name="three_component_9_tray_column",
        n_equilibrium_trays=9,
        stage_pressures_pa=np.linspace(180e3, 205e3, 11),
    )
    result = simulate(expanded).results[0]
    assert result.temperature_k.shape == (11,)
    assert result.liquid_composition.shape == (11, 3)


def test_nonconvergence_has_context(hydrocarbon_config):
    config = hydrocarbon_config.with_reflux_ratios([1.8])
    config = replace(config, solver=replace(config.solver, max_iterations=1, temperature_tau_tolerance_k2=1e-30))
    with pytest.raises(ConvergenceError, match=r"after 1 iterations.*worst stage=.*max \|dT\|"):
        ColumnSolver(config).solve(1.8)


def test_cli_writes_machine_readable_results_and_plots(project_root, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "distillation.cli",
            "simulate",
            str(project_root / "examples" / "three_component_column.yaml"),
            "--output",
            str(tmp_path),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "summary.json").is_file()
    assert list(tmp_path.glob("R_*.npz"))
    assert len(list((tmp_path / "plots").glob("*.png"))) == 5
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["runs"][0]["converged"]
