"""Patch 3 scientific integrity tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "threshold_capacity"))

from decision_model import (  # noqa: E402
    GenerativeParams,
    apply_temperature,
    evaluate_policy,
    expected_benefit,
    generate_base_dataset,
    policies_use_only_predictions,
    predictive_metrics,
    select_for_review,
    total_system_cost,
)


def test_cost_aware_policy_never_uses_realized_label():
    df = generate_base_dataset(0, GenerativeParams(n=3000))
    df = apply_temperature(df, 1.0)
    policies_use_only_predictions(df, 150)


def test_temperature_scaling_preserves_rank():
    base = generate_base_dataset(1, GenerativeParams(n=5000))
    p1 = apply_temperature(base, 1.0)["p_hat"].to_numpy()
    p2 = apply_temperature(base, 2.5)["p_hat"].to_numpy()
    assert np.array_equal(np.argsort(-p1), np.argsort(-p2))


def test_t1_calibrated_by_construction_bins():
    """In large samples, mean y ≈ mean p_base within loose tolerance."""
    df = generate_base_dataset(2, GenerativeParams(n=50_000))
    # overall calibration
    assert abs(df["y"].mean() - df["p_base"].mean()) < 0.01
    # coarse bin check
    bins = np.linspace(0, 1, 6)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df["p_base"] >= lo) & (df["p_base"] < hi)
        if m.sum() < 200:
            continue
        assert abs(df.loc[m, "y"].mean() - df.loc[m, "p_base"].mean()) < 0.05


def test_d1_fills_capacity():
    df = apply_temperature(generate_base_dataset(3, GenerativeParams(n=1000)), 1.0)
    m = select_for_review(df, "D1_cost_aware_topk", 100)
    assert m.sum() == 100


def test_d2_never_selects_nonpositive_benefit():
    df = apply_temperature(generate_base_dataset(4, GenerativeParams(n=5000)), 1.0)
    m = select_for_review(df, "D2_cost_aware_abstention", 200)
    assert m.sum() <= 200
    b = expected_benefit(
        df["p_hat"].to_numpy(),
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        25.0,
        1.0,
    )
    assert (b[m] > 0).all()


def test_d1_d2_identical_ranking_order():
    df = apply_temperature(generate_base_dataset(5, GenerativeParams(n=3000)), 1.0)
    b = expected_benefit(
        df["p_hat"].to_numpy(),
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        25.0,
        1.0,
    )
    order = np.argsort(-b, kind="mergesort")
    m1 = select_for_review(df, "D1_cost_aware_topk", 100)
    # D1 top-100 equals first 100 of benefit order
    assert set(np.flatnonzero(m1)) == set(order[:100].tolist())


def test_homogeneous_positive_ev_masks_identical():
    params = GenerativeParams(n=10_000, constant_value=500.0, constant_fp=20.0)
    df = apply_temperature(generate_base_dataset(42, params), 1.0)
    k = 300
    b = expected_benefit(
        df["p_hat"].to_numpy(),
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        25.0,
        1.0,
    )
    top = np.argsort(-df["p_hat"].to_numpy())[:k]
    assert (b[top] > 0).all()
    mb = select_for_review(df, "B_topk_probability", k)
    mc = select_for_review(df, "C_expected_exposure", k)
    md1 = select_for_review(df, "D1_cost_aware_topk", k)
    md2 = select_for_review(df, "D2_cost_aware_abstention", k)
    assert np.array_equal(mb, mc)
    assert np.array_equal(mb, md1)
    assert np.array_equal(mb, md2)


def test_review_effectiveness_hand_calc():
    y = np.array([1, 0, 1])
    v = np.array([100.0, 50.0, 80.0])
    f = np.array([10.0, 10.0, 10.0])
    q = np.array([True, True, False])
    r = 5.0
    e = 0.5
    # unreviewed fraud: case2 loss 80
    # residual reviewed fraud: case0 * 0.5 * 100 = 50
    # review cost 10; FP case1 = 10
    got = total_system_cost(y, v, f, q, r, e)
    assert got["fraud_loss_remaining"] == 130.0
    assert got["review_cost"] == 10.0
    assert got["fp_intervention_cost"] == 10.0
    assert got["total_system_cost"] == 150.0
    assert abs(expected_benefit(np.array([0.5]), np.array([100.0]), np.array([10.0]), 5.0, 0.5)[0] - (0.5 * 0.5 * 100 - 5 - 0.5 * 10)) < 1e-9


def test_determinism():
    p = GenerativeParams(n=2000)
    a = generate_base_dataset(99, p)
    b = generate_base_dataset(99, p)
    pd.testing.assert_frame_equal(a, b)


def test_auc_stable_across_T():
    base = generate_base_dataset(7, GenerativeParams(n=20_000))
    m1 = predictive_metrics(base["y"].to_numpy(), apply_temperature(base, 1.0)["p_hat"].to_numpy())
    m2 = predictive_metrics(base["y"].to_numpy(), apply_temperature(base, 2.5)["p_hat"].to_numpy())
    assert abs(m1["roc_auc"] - m2["roc_auc"]) < 1e-9
    assert m1["brier"] != m2["brier"]
