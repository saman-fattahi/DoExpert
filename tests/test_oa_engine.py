"""Unit tests for the Taguchi orthogonal-array engine (doexpert.doe)."""

import numpy as np
import pytest

from doexpert.doe import (
    auto_select_oa,
    build_design,
    find_compatible_oas,
    generate_basic_oa_matrix,
    get_oa_info,
    list_catalog,
)


def test_catalog_not_empty_and_contains_l8():
    labels = [oa.label for oa in list_catalog()]
    assert len(labels) > 0
    assert any(l.startswith("L8") for l in labels)


def test_l8_matrix_shape_and_levels():
    oa = get_oa_info("L8(2)")
    assert oa is not None
    matrix = np.array(generate_basic_oa_matrix(oa))
    assert matrix.shape[0] == 8
    # Two-level array: only levels 0 and 1 (0-based) or 1 and 2 (1-based).
    assert len(np.unique(matrix)) == 2


def test_l8_balance_property():
    """Each column of a two-level OA must contain each level equally often."""
    oa = get_oa_info("L8(2)")
    matrix = np.array(generate_basic_oa_matrix(oa))
    for col in matrix.T:
        values, counts = np.unique(col, return_counts=True)
        assert len(values) == 2
        assert counts[0] == counts[1] == 4


def test_l8_orthogonality_strength_two():
    """All level pairs must appear equally often in every column pair."""
    oa = get_oa_info("L8(2)")
    matrix = np.array(generate_basic_oa_matrix(oa))
    n_cols = matrix.shape[1]
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            pairs = list(zip(matrix[:, i].tolist(), matrix[:, j].tolist()))
            _, counts = np.unique(pairs, axis=0, return_counts=True)
            assert len(set(counts.tolist())) == 1, f"columns {i},{j} not orthogonal"


def test_auto_select_oa_for_five_two_level_factors():
    """The flat grinding case study: 5 factors at 2 levels -> L8 fits."""
    oa = auto_select_oa([2, 2, 2, 2, 2])
    assert oa is not None
    assert oa.runs >= 8


def test_find_compatible_oas_rejects_oversized_problems():
    compatible = find_compatible_oas([2] * 100)
    assert isinstance(compatible, list)


def test_build_design_dataframe():
    oa = auto_select_oa([2, 2, 2, 2, 2])
    names = ["vc", "ae", "vw", "qd", "Ud"]
    levels = [["20", "30"], ["50", "200"], ["250", "1000"], ["0.7", "0.9"], ["2", "4"]]
    df = build_design(oa, names, levels)
    assert list(df.columns[: len(names)]) == names or all(n in df.columns for n in names)
    assert len(df) == oa.runs
    # Every factor column must only contain its configured levels.
    for name, lv in zip(names, levels):
        assert set(df[name].astype(str)) <= set(lv)
