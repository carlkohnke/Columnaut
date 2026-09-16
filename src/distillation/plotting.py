"""Stage-profile and operating-specification plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .results import SimulationBatch, SimulationResult


def _save_profile(result: SimulationResult, output: Path, quantity: str) -> Path:
    stages = result.stage_numbers
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    if quantity == "liquid_composition":
        for i, name in enumerate(result.component_names):
            ax.plot(stages, result.liquid_composition[:, i], label=name)
        ax.set_ylabel("Liquid mole fraction, x")
        ax.set_ylim(0, 1)
        title = "Liquid composition versus engineering stage"
    elif quantity == "vapor_composition":
        for i, name in enumerate(result.component_names):
            ax.plot(stages, result.vapor_composition[:, i], label=name)
        ax.set_ylabel("Vapor mole fraction, y")
        ax.set_ylim(0, 1)
        title = "Vapor composition versus engineering stage"
    elif quantity == "temperature":
        ax.plot(stages, result.temperature_k, color="tab:red")
        ax.set_ylabel("Temperature (K)")
        title = "Temperature versus engineering stage"
    elif quantity == "flows":
        ax.plot(stages, result.liquid_flow_kmol_h, label="Liquid L")
        ax.plot(stages, result.vapor_flow_kmol_h, label="Vapor V")
        ax.set_ylabel("Molar flow (kmol/h)")
        title = "Internal liquid and vapor flow versus engineering stage"
    else:
        raise ValueError(quantity)
    ax.set_xlabel("Engineering stage (1 = total condenser; last = partial reboiler)")
    ax.set_xlim(stages[0], stages[-1])
    ax.grid(alpha=0.25)
    ax.set_title(f"{title}\nR = {result.reflux_ratio:g}")
    if quantity in {"liquid_composition", "vapor_composition", "flows"}:
        ax.legend(ncol=2, fontsize="small")
    path = output / f"R_{result.reflux_ratio:.1f}_{quantity}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_batch(batch: SimulationBatch, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for result in batch.results:
        for quantity in ("liquid_composition", "vapor_composition", "temperature", "flows"):
            paths.append(_save_profile(result, output, quantity))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ratios = [r.reflux_ratio for r in batch.results]
    condenser = [r.reported_condenser_duty_kj_h for r in batch.results]
    reboiler = [r.reported_reboiler_duty_kj_h for r in batch.results]
    ax.plot(ratios, condenser, marker="o", label="Condenser")
    ax.plot(ratios, reboiler, marker="o", label="Reboiler")
    ax.set_xlabel("Reflux ratio L/D")
    ax.set_ylabel("Reported duty (kJ/h)")
    ax.set_title("Duty versus reflux ratio")
    ax.grid(alpha=0.25)
    ax.legend()
    path = output / "duty_vs_reflux_ratio.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths
