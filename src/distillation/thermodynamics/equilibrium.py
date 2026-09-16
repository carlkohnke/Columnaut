"""Robust scalar equilibrium-temperature root solving."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import brentq

from .base import ThermodynamicError


def bracketed_temperature_root(
    residual: Callable[[float], float],
    lower_k: float,
    upper_k: float,
    *,
    xtol: float = 1e-10,
    initial_k: float | None = None,
) -> float:
    """Find a finite sign-changing bracket on a temperature scan, then use Brent."""

    if initial_k is not None:
        center = float(np.clip(initial_k, lower_k, upper_k))
        try:
            center_value = float(residual(center))
        except (ThermodynamicError, FloatingPointError, ValueError):
            center_value = np.nan
        if np.isfinite(center_value) and abs(center_value) < 1e-14:
            return center
        step = max(1.0, 0.01 * center)
        for _ in range(18):
            low = max(lower_k, center - step)
            high = min(upper_k, center + step)
            try:
                low_value = float(residual(low))
            except (ThermodynamicError, FloatingPointError, ValueError):
                low_value = np.nan
            try:
                high_value = float(residual(high))
            except (ThermodynamicError, FloatingPointError, ValueError):
                high_value = np.nan
            candidates: list[tuple[float, float]] = []
            if np.isfinite(low_value) and np.isfinite(center_value) and low_value * center_value < 0:
                candidates.append((low, center))
            if np.isfinite(center_value) and np.isfinite(high_value) and center_value * high_value < 0:
                candidates.append((center, high))
            if np.isfinite(low_value) and np.isfinite(high_value) and low_value * high_value < 0:
                candidates.append((low, high))
            if candidates:
                chosen = min(candidates, key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - center))
                return float(
                    brentq(residual, chosen[0], chosen[1], xtol=xtol, rtol=1e-12, maxiter=200)
                )
            if low == lower_k and high == upper_k:
                break
            step *= 1.7

    grid = np.linspace(lower_k, upper_k, 160)
    last_t: float | None = None
    last_r: float | None = None
    brackets: list[tuple[float, float]] = []
    exact: list[float] = []
    failures: list[str] = []
    for t in grid:
        try:
            value = float(residual(float(t)))
        except (ThermodynamicError, FloatingPointError, ValueError) as exc:
            failures.append(f"{t:.3f} K: {exc}")
            continue
        if not np.isfinite(value):
            continue
        if value == 0.0:
            exact.append(float(t))
        if last_r is not None and value * last_r < 0.0:
            brackets.append((float(last_t), float(t)))
        last_t, last_r = float(t), value
    if brackets:
        if initial_k is None:
            chosen = brackets[0]
        else:
            chosen = min(brackets, key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - initial_k))
        return float(brentq(residual, chosen[0], chosen[1], xtol=xtol, rtol=1e-12, maxiter=200))
    if exact:
        # A one-real-root EOS state with x == y can make K == 1 over a broad
        # range; prefer an actual sign-changing phase-equilibrium root above.
        if initial_k is None:
            return exact[0]
        return min(exact, key=lambda value: abs(value - initial_k))
    detail = failures[-1] if failures else "no finite sign change"
    raise ThermodynamicError(
        f"Could not bracket an equilibrium temperature in [{lower_k}, {upper_k}] K ({detail})"
    )
