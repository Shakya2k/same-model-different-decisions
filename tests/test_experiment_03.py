"""Tests for Experiment 03 — fair information-set misspecification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "threshold_capacity"))
sys.path.insert(0, str(ROOT / "experiments" / "decision_misspecification"))

from decision_model import GenerativeParams, apply_temperature, generate_base_dataset  # noqa: E402
from estimation import (  # noqa: E402
    SIGMA_VALUE_HIGH,
    EstimateBundle,
    build_estimates,
    estimated_benefit_scores,
    evaluate_mask_true,
    multiplicative_lognormal_noise,
    oracle_benefit_scores,
    run_policy_row,
    select_with_quantities,
)


def _tiny_df(seed: int = 0, n: int = 800):
    base = generate_base_dataset(seed, GenerativeParams(n=n))
    return apply_temperature(base, 1.0)


def test_evaluation_uses_true_quantities_only():
    df = _tiny_df(1)
    est = build_estimates(
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        0.8,
        25.0,
        "VALUE_NOISE_HIGH",
        1,
    )
    mask = select_with_quantities(
        df, "D1_cost_aware_topk", 50, est.value_hat, est.fp_hat, est.r_hat, est.e_hat
    )
    from decision_model import total_system_cost

    m = evaluate_mask_true(df, mask, e_true=0.8, r_true=25.0)
    m2 = total_system_cost(
        df["y"].to_numpy(),
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        mask,
        25.0,
        0.8,
    )
    assert m["total_system_cost"] == m2["total_system_cost"]


def test_equal_information_sets_share_value_estimates():
    df = _tiny_df(2, n=2000)
    est = build_estimates(
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        0.8,
        25.0,
        "VALUE_NOISE_HIGH",
        2,
    )
    r_c = run_policy_row(df, "C_est", 40, 0.8, 25.0, est)
    r_d = run_policy_row(df, "D1_est", 40, 0.8, 25.0, est)
    assert r_c["condition"] == r_d["condition"] == "VALUE_NOISE_HIGH"
    r_co = run_policy_row(df, "C_oracle", 40, 0.8, 25.0, est)
    assert not np.isclose(r_c["total_system_cost"], r_co["total_system_cost"]) or (
        r_c["review_volume"] == r_co["review_volume"]
    )
    mask_est = select_with_quantities(
        df, "C_expected_exposure", 40, est.value_hat, df["fp_cost"].to_numpy(), 25.0, 1.0
    )
    mask_true = select_with_quantities(
        df,
        "C_expected_exposure",
        40,
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        25.0,
        1.0,
    )
    assert not np.array_equal(mask_est, mask_true)


def test_c_est_does_not_use_true_v_under_value_noise():
    """Fairness check: C_est ranks on v_hat, not true v."""
    df = _tiny_df(22, n=1500)
    est = build_estimates(
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        0.8,
        25.0,
        "VALUE_NOISE_LOW",
        22,
    )
    assert not np.allclose(est.value_hat, df["value"].to_numpy())
    m_est = select_with_quantities(
        df, "C_expected_exposure", 30, est.value_hat, df["fp_cost"].to_numpy(), 25.0, 1.0
    )
    m_true = select_with_quantities(
        df,
        "C_expected_exposure",
        30,
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        25.0,
        1.0,
    )
    assert not np.array_equal(m_est, m_true)


def test_oracle_and_estimated_policy_match_without_noise():
    df = _tiny_df(3)
    est = build_estimates(
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        0.8,
        25.0,
        "ORACLE",
        3,
    )
    assert np.allclose(est.value_hat, df["value"].to_numpy())
    assert est.e_hat == 0.8
    assert np.allclose(
        estimated_benefit_scores(df, est), oracle_benefit_scores(df, 0.8, 25.0)
    )
    assert (
        run_policy_row(df, "D1_oracle", 30, 0.8, 25.0, est)["total_system_cost"]
        == run_policy_row(df, "D1_est", 30, 0.8, 25.0, est)["total_system_cost"]
    )
    assert (
        run_policy_row(df, "C_oracle", 30, 0.8, 25.0, est)["total_system_cost"]
        == run_policy_row(df, "C_est", 30, 0.8, 25.0, est)["total_system_cost"]
    )


def test_mean_one_multiplicative_noise_approximately_unbiased():
    rng = np.random.default_rng(12345)
    x = np.full(200_000, 100.0)
    xhat = multiplicative_lognormal_noise(x, rng, SIGMA_VALUE_HIGH, mean_one=True)
    assert abs(float(np.mean(xhat / x)) - 1.0) < 0.01


def test_estimates_remain_positive():
    df = _tiny_df(4, n=2000)
    for cond in ("VALUE_NOISE_HIGH", "FP_COST_NOISE_HIGH", "JOINT_MODERATE_MISSPECIFICATION"):
        est = build_estimates(
            df["value"].to_numpy(), df["fp_cost"].to_numpy(), 0.8, 25.0, cond, 4
        )
        assert np.all(est.value_hat > 0)
        assert np.all(est.fp_hat > 0)


def test_same_seed_same_estimates():
    df = _tiny_df(5)
    a = build_estimates(
        df["value"].to_numpy(), df["fp_cost"].to_numpy(), 0.8, 25.0, "VALUE_NOISE_LOW", 5
    )
    b = build_estimates(
        df["value"].to_numpy(), df["fp_cost"].to_numpy(), 0.8, 25.0, "VALUE_NOISE_LOW", 5
    )
    assert np.array_equal(a.value_hat, b.value_hat)


def test_policy_selection_never_accesses_y():
    df = _tiny_df(6)
    est = build_estimates(
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        0.8,
        25.0,
        "JOINT_MODERATE_MISSPECIFICATION",
        6,
    )
    df_flip = df.copy()
    df_flip["y"] = 1 - df_flip["y"]
    for pol in ("D1_cost_aware_topk", "D2_cost_aware_abstention", "C_expected_exposure"):
        m1 = select_with_quantities(
            df, pol, 25, est.value_hat, est.fp_hat, est.r_hat, est.e_hat
        )
        m2 = select_with_quantities(
            df_flip, pol, 25, est.value_hat, est.fp_hat, est.r_hat, est.e_hat
        )
        assert np.array_equal(m1, m2)


def test_paired_comparison_same_dataset_seed():
    df = _tiny_df(7, n=1500)
    est = build_estimates(
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        0.8,
        25.0,
        "VALUE_NOISE_HIGH",
        7,
    )
    r_b = run_policy_row(df, "B_topk_probability", 40, 0.8, 25.0, est)
    r_c = run_policy_row(df, "C_est", 40, 0.8, 25.0, est)
    r_d = run_policy_row(df, "D1_est", 40, 0.8, 25.0, est)
    assert r_b["condition"] == r_c["condition"] == r_d["condition"]
    assert all(np.isfinite(x["total_system_cost"]) for x in (r_b, r_c, r_d))


def test_extreme_misspecification_no_nan_scores():
    df = _tiny_df(8, n=1000)
    est = build_estimates(
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        0.8,
        25.0,
        "VALUE_NOISE_HIGH",
        8,
    )
    wild = EstimateBundle(
        value_hat=est.value_hat * 1e6,
        fp_hat=est.fp_hat * 1e6,
        e_hat=1.0,
        r_hat=25.0,
        condition="VALUE_NOISE_HIGH",
    )
    assert np.all(np.isfinite(estimated_benefit_scores(df, wild)))
    row = run_policy_row(df, "D1_est", 20, 0.8, 25.0, wild)
    assert np.isfinite(row["total_system_cost"])
