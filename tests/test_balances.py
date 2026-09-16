from __future__ import annotations

import numpy as np

from distillation.balances import solve_lower_bidiagonal, solve_tridiagonal


def test_tridiagonal_solver_matches_numpy():
    rng = np.random.default_rng(42)
    n = 20
    lower = rng.uniform(-0.3, 0.3, n - 1)
    upper = rng.uniform(-0.3, 0.3, n - 1)
    diagonal = rng.uniform(2.0, 3.0, n)
    rhs = rng.normal(size=n)
    matrix = np.diag(diagonal) + np.diag(lower, -1) + np.diag(upper, 1)
    assert np.allclose(solve_tridiagonal(lower, diagonal, upper, rhs), np.linalg.solve(matrix, rhs))


def test_lower_bidiagonal_solver_matches_numpy():
    rng = np.random.default_rng(7)
    n = 15
    sub = rng.uniform(-1, 1, n - 1)
    diagonal = rng.uniform(1.5, 3, n)
    rhs = rng.normal(size=n)
    matrix = np.diag(diagonal) + np.diag(sub, -1)
    assert np.allclose(solve_lower_bidiagonal(sub, diagonal, rhs), np.linalg.solve(matrix, rhs))
