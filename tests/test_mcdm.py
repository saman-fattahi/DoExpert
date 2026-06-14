"""Unit tests for doexpert.mcdm.

Each MCDM method is tested against properties that any valid ranking
procedure must satisfy on simple, analytically tractable decision
matrices: (i) a strictly dominating alternative is ranked first,
(ii) score arrays have the right shape and finite values,
(iii) the unified dispatcher applies the higher-is-better convention
consistently across methods.
"""

import numpy as np
import pytest

from doexpert.mcdm import (
    AVAILABLE_METHODS,
    fuzzy_topsis,
    promethee_ii,
    rank_solutions,
    saw,
    topsis,
    vikor,
    waspas,
)

# Three alternatives, two criteria. Alternative 0 dominates on both
# criteria when c0 is 'max' and c1 is 'min'.
DOMINANT = np.array(
    [
        [10.0, 1.0],  # best on both
        [5.0, 5.0],
        [1.0, 10.0],  # worst on both
    ]
)
WEIGHTS = np.array([0.5, 0.5])
TYPES = ["max", "min"]


@pytest.mark.parametrize("method", [topsis, saw, waspas, promethee_ii, fuzzy_topsis])
def test_dominant_alternative_ranked_first(method):
    result = method(DOMINANT, WEIGHTS, TYPES)
    scores, ranking = result[0], result[1]
    assert ranking[0] == 0, f"{method.__name__} failed to rank dominant alternative first"
    assert ranking[-1] == 2
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_vikor_dominant_alternative_ranked_first():
    result = vikor(DOMINANT, WEIGHTS, TYPES, v=0.5)
    scores, ranking, Q = result[0], result[1], result[2]
    # Returned scores follow the higher-is-better convention (1 - Q);
    # the raw VIKOR index Q (lower is better) is returned as diagnostic.
    assert ranking[0] == 0
    assert scores[0] == np.max(scores)
    assert Q[0] == np.min(Q)


def test_topsis_scores_bounded():
    scores, _ = topsis(DOMINANT, WEIGHTS, TYPES)[:2]
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_saw_equal_alternatives_get_equal_scores():
    m = np.array([[2.0, 3.0], [2.0, 3.0]])
    scores = saw(m, WEIGHTS, TYPES)[0]
    assert scores[0] == pytest.approx(scores[1])


@pytest.mark.parametrize("name", sorted(AVAILABLE_METHODS))
def test_dispatcher_higher_is_better_for_all_methods(name):
    scores, ranking = rank_solutions(DOMINANT, WEIGHTS, TYPES, method=name)
    # After harmonisation, the best alternative must hold the highest score.
    assert ranking[0] == 0
    assert scores[ranking[0]] == pytest.approx(np.max(scores))


def test_dispatcher_normalises_weights():
    s1, r1 = rank_solutions(DOMINANT, np.array([1.0, 1.0]), TYPES, method="TOPSIS")
    s2, r2 = rank_solutions(DOMINANT, np.array([10.0, 10.0]), TYPES, method="TOPSIS")
    np.testing.assert_allclose(s1, s2)
    np.testing.assert_array_equal(r1, r2)


def test_dispatcher_rejects_unknown_method():
    with pytest.raises(ValueError):
        rank_solutions(DOMINANT, WEIGHTS, TYPES, method="NOT_A_METHOD")


def test_weight_sensitivity_changes_ranking():
    # Alternative 0 excels on criterion 0, alternative 1 on criterion 1.
    m = np.array([[10.0, 10.0], [1.0, 1.0]])
    types = ["max", "min"]
    _, rank_c0 = rank_solutions(m, np.array([1.0, 0.0]), types, method="TOPSIS")
    _, rank_c1 = rank_solutions(m, np.array([0.0, 1.0]), types, method="TOPSIS")
    assert rank_c0[0] == 0  # weight on 'max' criterion favours alt 0
    assert rank_c1[0] == 1  # weight on 'min' criterion favours alt 1
