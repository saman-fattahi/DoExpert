"""Unit tests for doexpert.metrics."""

import numpy as np
import pytest

from doexpert.metrics import (
    calculate_extent_metric,
    calculate_hypervolume,
    calculate_spacing_metric,
)


def test_hypervolume_2d_known_value():
    # Single point (0.5, 0.5) with reference (1, 1): HV = 0.25.
    front = np.array([[0.5, 0.5]])
    hv = calculate_hypervolume(front, reference_point=np.array([1.0, 1.0]))
    assert hv == pytest.approx(0.25, rel=0.05)


def test_hypervolume_monotone_in_front_quality():
    ref = np.array([1.0, 1.0])
    better = np.array([[0.2, 0.2]])
    worse = np.array([[0.8, 0.8]])
    assert calculate_hypervolume(better, reference_point=ref) > calculate_hypervolume(
        worse, reference_point=ref
    )


def test_spacing_uniform_front_is_zero():
    # Equally spaced points -> spacing metric 0.
    front = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    assert calculate_spacing_metric(front) == pytest.approx(0.0, abs=1e-9)


def test_spacing_nonuniform_front_positive():
    front = np.array([[0.0, 3.0], [0.1, 2.9], [3.0, 0.0]])
    assert calculate_spacing_metric(front) > 0.0


def test_extent_metric_increases_with_spread():
    narrow = np.array([[0.0, 0.0], [0.1, 0.1]])
    wide = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert calculate_extent_metric(wide) > calculate_extent_metric(narrow)
