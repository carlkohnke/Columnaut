"""Structured simulation results, diagnostics, and serialization."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(slots=True)
class SimulationResult:
    name: str
    reflux_ratio: float
    component_names: tuple[str, ...]
    stage_numbers: NDArray[np.int64]
    pressure_pa: NDArray[np.float64]
    temperature_k: NDArray[np.float64]
    liquid_composition: NDArray[np.float64]
    vapor_composition: NDArray[np.float64]
    k_values: NDArray[np.float64]
    liquid_flow_kmol_h: NDArray[np.float64]
    vapor_flow_kmol_h: NDArray[np.float64]
    feed_flow_kmol_h: NDArray[np.float64]
    distillate_flow_kmol_h: float
    bottoms_flow_kmol_h: float
    condenser_duty_rate_kj_h: float
    reboiler_duty_rate_kj_h: float
    reported_condenser_duty_kj_h: float
    reported_reboiler_duty_kj_h: float
    iterations: int
    converged: bool
    solve_time_seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        stem = f"R_{self.reflux_ratio:.1f}"
        with (output / f"{stem}.json").open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, allow_nan=False)
        np.savez_compressed(
            output / f"{stem}.npz",
            stage_numbers=self.stage_numbers,
            pressure_pa=self.pressure_pa,
            temperature_k=self.temperature_k,
            x=self.liquid_composition,
            y=self.vapor_composition,
            K=self.k_values,
            L_kmol_h=self.liquid_flow_kmol_h,
            V_kmol_h=self.vapor_flow_kmol_h,
            feed_kmol_h=self.feed_flow_kmol_h,
        )
        with (output / f"{stem}_stages.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            columns = ["stage", "pressure_kPa", "temperature_K", "L_kmol_h", "V_kmol_h"]
            for phase in ("x", "y", "K"):
                columns.extend(f"{phase}_{name}" for name in self.component_names)
            writer.writerow(columns)
            for i, stage in enumerate(self.stage_numbers):
                writer.writerow(
                    [
                        int(stage),
                        self.pressure_pa[i] / 1e3,
                        self.temperature_k[i],
                        self.liquid_flow_kmol_h[i],
                        self.vapor_flow_kmol_h[i],
                        *self.liquid_composition[i],
                        *self.vapor_composition[i],
                        *self.k_values[i],
                    ]
                )


@dataclass(slots=True)
class SimulationBatch:
    results: list[SimulationResult]
    total_solve_time_seconds: float

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for result in self.results:
            result.save(output)
        summary = {
            "total_solve_time_seconds": self.total_solve_time_seconds,
            "runs": [
                {
                    "reflux_ratio": r.reflux_ratio,
                    "iterations": r.iterations,
                    "converged": r.converged,
                    "solve_time_seconds": r.solve_time_seconds,
                    "condenser_duty_rate_kj_h": r.condenser_duty_rate_kj_h,
                    "reboiler_duty_rate_kj_h": r.reboiler_duty_rate_kj_h,
                    "reported_condenser_duty_kj_h": r.reported_condenser_duty_kj_h,
                    "reported_reboiler_duty_kj_h": r.reported_reboiler_duty_kj_h,
                    "max_component_balance_residual_kmol_h": r.diagnostics.get(
                        "max_component_balance_residual_kmol_h"
                    ),
                    "max_bubble_point_residual": r.diagnostics.get("max_bubble_point_residual"),
                }
                for r in self.results
            ],
        }
        with (output / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(summary), stream, indent=2, allow_nan=False)
