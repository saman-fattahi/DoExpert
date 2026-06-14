"""Unit tests for doexpert.pareto."""

import numpy as np

from doexpert.pareto import get_pareto, is_pareto_efficient, is_pareto_optimal


def test_is_pareto_efficient_simple_min():
    # Minimisation on both objectives: points on the line x+y=2 are
    # mutually non-dominated; (2, 2) is dominated by (1, 1).
    costs = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [2.0, 2.0]])
    mask = is_pareto_efficient(costs, return_mask=True)
    assert mask.tolist() == [True, True, True, False]


def test_is_pareto_optimal_with_maximize_flags():
    # Objective 0 minimised, objective 1 maximised.
    costs = np.array([[1.0, 10.0], [1.0, 5.0], [2.0, 10.0]])
    mask = is_pareto_optimal(costs, maximize_objectives=[False, True])
    # Point 0 dominates both others.
    assert mask[0]
    assert not mask[1]
    assert not mask[2]


def test_get_pareto_returns_consistent_subsets():
    rng = np.random.default_rng(42)
    X = rng.random((50, 3))
    F = rng.random((50, 2))
    result = get_pareto(X, F, maximize_objectives=[False, False], return_indices=True)
    idx = result["indices"]
    assert len(result["X_nd"]) == len(result["F_nd"]) == len(idx)
    assert len(idx) >= 1
    # Every returned point must be non-dominated within the full set.
    full_mask = is_pareto_optimal(F, maximize_objectives=[False, False])
    assert all(full_mask[i] for i in idx)


def test_single_point_is_pareto():
    F = np.array([[1.0, 2.0]])
    mask = is_pareto_efficient(F, return_mask=True)
    assert mask.tolist() == [True]
