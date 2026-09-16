"""Command-line interface for simulations and result generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .configuration import load_config
from .plotting import plot_batch
from .solver import ConvergenceError, simulate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distillation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("simulate", help="simulate a YAML or JSON column configuration")
    run.add_argument("configuration", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--reflux-ratio", type=float, action="append", dest="reflux_ratios")
    run.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "simulate":
        config = load_config(args.configuration)
        if args.reflux_ratios:
            config = config.with_reflux_ratios(args.reflux_ratios)
        try:
            batch = simulate(config)
        except (ConvergenceError, ValueError, RuntimeError) as exc:
            print(f"Simulation failed: {exc}", file=sys.stderr)
            return 2
        batch.save(args.output)
        if not args.no_plots:
            plot_batch(batch, args.output / "plots")
        print(f"Completed {len(batch.results)} run(s) in {batch.total_solve_time_seconds:.6f} s")
        for result in batch.results:
            print(
                f"R={result.reflux_ratio:g} iterations={result.iterations} "
                f"tau={result.diagnostics['temperature_tau_history_k2'][-1]:.3e} K^2 "
                f"max|bubble|={result.diagnostics['max_bubble_point_residual']:.3e} "
                f"Qc={result.reported_condenser_duty_kj_h:.6g} kJ/h "
                f"Qr={result.reported_reboiler_duty_kj_h:.6g} kJ/h"
            )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
