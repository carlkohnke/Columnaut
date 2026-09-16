"""Banded linear algebra for material and internal-flow balances."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded


class BalanceSolveError(RuntimeError):
    """A material- or energy-balance matrix was singular or invalid."""


def solve_tridiagonal(
    lower: ArrayLike, diagonal: ArrayLike, upper: ArrayLike, rhs: ArrayLike
) -> NDArray[np.float64]:
    lower_a = np.asarray(lower, dtype=float)
    diagonal_a = np.asarray(diagonal, dtype=float)
    upper_a = np.asarray(upper, dtype=float)
    rhs_a = np.asarray(rhs, dtype=float)
    n = diagonal_a.size
    if lower_a.shape != (n - 1,) or upper_a.shape != (n - 1,) or rhs_a.shape != (n,):
        raise BalanceSolveError(
            f"Tridiagonal dimensions must be ({n-1}, {n}, {n-1}, {n}); got "
            f"{lower_a.shape}, {diagonal_a.shape}, {upper_a.shape}, {rhs_a.shape}"
        )
    ab = np.zeros((3, n), dtype=float)
    ab[0, 1:] = upper_a
    ab[1] = diagonal_a
    ab[2, :-1] = lower_a
    try:
        result = solve_banded((1, 1), ab, rhs_a, check_finite=True)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise BalanceSolveError(f"Tridiagonal component balance failed: {exc}") from exc
    if not np.all(np.isfinite(result)):
        raise BalanceSolveError("Tridiagonal component balance returned non-finite values")
    return result


def solve_lower_bidiagonal(
    subdiagonal: ArrayLike, diagonal: ArrayLike, rhs: ArrayLike
) -> NDArray[np.float64]:
    sub = np.asarray(subdiagonal, dtype=float)
    diag = np.asarray(diagonal, dtype=float)
    rhs_a = np.asarray(rhs, dtype=float)
    n = diag.size
    if sub.shape != (n - 1,) or rhs_a.shape != (n,):
        raise BalanceSolveError(
            f"Lower-bidiagonal dimensions must be ({n-1}, {n}, {n}); got {sub.shape}, {diag.shape}, {rhs_a.shape}"
        )
    if np.any(diag == 0.0):
        raise BalanceSolveError("Lower-bidiagonal energy balance has a zero diagonal")
    result = np.empty(n, dtype=float)
    result[0] = rhs_a[0] / diag[0]
    for i in range(1, n):
        result[i] = (rhs_a[i] - sub[i - 1] * result[i - 1]) / diag[i]
    if not np.all(np.isfinite(result)):
        raise BalanceSolveError("Lower-bidiagonal energy balance returned non-finite values")
    return result


def tridiagonal_residual(
    lower: ArrayLike, diagonal: ArrayLike, upper: ArrayLike, x: ArrayLike, rhs: ArrayLike
) -> NDArray[np.float64]:
    lower_a = np.asarray(lower, dtype=float)
    diagonal_a = np.asarray(diagonal, dtype=float)
    upper_a = np.asarray(upper, dtype=float)
    x_a = np.asarray(x, dtype=float)
    value = diagonal_a * x_a
    value[1:] += lower_a * x_a[:-1]
    value[:-1] += upper_a * x_a[1:]
    return value - np.asarray(rhs, dtype=float)
