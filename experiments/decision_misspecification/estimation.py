"""
Fair misspecification helpers for Experiment 03.

Evaluation uses true (v, f, e, r). Deployable ranking may use estimates.
C_oracle / C_est score with p·v / p·v_hat (e not in the C information set).
Under value noise, C_est and D*_est share the same v_hat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

import sys
from pathlib import Path

_TC = Path(__file__).resolve().parents[1] / "threshold_capacity"
if str(_TC) not in sys.path:
    sys.path.insert(0, str(_TC))

from decision_model import (  # noqa: E402
    REVIEW_COST_PER_CASE,
    PolicyName,
    expected_benefit,
    select_for_review,
    total_system_cost,
)

ConditionName = Literal[
    "ORACLE",
    "VALUE_NOISE_LOW",
    "VALUE_NOISE_HIGH",
    "FP_COST_NOISE_LOW",
    "FP_COST_NOISE_HIGH",
    "REVIEW_EFFECTIVENESS_OPTIMISM",
    "REVIEW_EFFECTIVENESS_PESSIMISM",
    "JOINT_MODERATE_MISSPECIFICATION",
]

SIGMA_VALUE_LOW = math.sqrt(math.log(1.0 + 0.20**2))
SIGMA_VALUE_HIGH = math.sqrt(math.log(1.0 + 0.50**2))
SIGMA_FP_LOW = SIGMA_VALUE_LOW
SIGMA_FP_HIGH = SIGMA_VALUE_HIGH
SIGMA_JOINT_V = math.sqrt(math.log(1.0 + 0.30**2))
SIGMA_JOINT_F = math.sqrt(math.log(1.0 + 0.30**2))

E_TRUE_DEFAULT = 0.8
E_HAT_OPTIMISTIC = 1.0
E_HAT_PESSIMISTIC = 0.6
E_HAT_JOINT = 0.9

# Fair information-set labels (publication)
PolicyLabel = Literal[
    "B_topk_probability",
    "C_oracle",
    "C_est",
    "D1_oracle",
    "D2_oracle",
    "D1_est",
    "D2_est",
]


@dataclass(frozen=True)
class EstimateBundle:
    value_hat: np.ndarray
    fp_hat: np.ndarray
    e_hat: float
    r_hat: float
    condition: ConditionName


def multiplicative_lognormal_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    mean_one: bool = True,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if sigma <= 0:
        return x.copy()
    mu = -0.5 * sigma * sigma if mean_one else 0.0
    eta = rng.normal(mu, sigma, size=x.shape)
    out = x * np.exp(eta)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite estimates from multiplicative noise")
    if np.any(out <= 0):
        raise ValueError("non-positive estimates from multiplicative noise")
    return out


def build_estimates(
    value: np.ndarray,
    fp_cost: np.ndarray,
    e_true: float,
    r_true: float,
    condition: ConditionName,
    seed: int,
) -> EstimateBundle:
    """Construct shared policy-facing estimates; never mutates true arrays."""
    rng = np.random.default_rng(seed + 17_003)
    v = np.asarray(value, dtype=float)
    f = np.asarray(fp_cost, dtype=float)
    e_hat = float(e_true)
    r_hat = float(r_true)
    v_hat = v.copy()
    f_hat = f.copy()

    if condition == "ORACLE":
        pass
    elif condition == "VALUE_NOISE_LOW":
        v_hat = multiplicative_lognormal_noise(v, rng, SIGMA_VALUE_LOW)
    elif condition == "VALUE_NOISE_HIGH":
        v_hat = multiplicative_lognormal_noise(v, rng, SIGMA_VALUE_HIGH)
    elif condition == "FP_COST_NOISE_LOW":
        f_hat = multiplicative_lognormal_noise(f, rng, SIGMA_FP_LOW)
    elif condition == "FP_COST_NOISE_HIGH":
        f_hat = multiplicative_lognormal_noise(f, rng, SIGMA_FP_HIGH)
    elif condition == "REVIEW_EFFECTIVENESS_OPTIMISM":
        e_hat = E_HAT_OPTIMISTIC
    elif condition == "REVIEW_EFFECTIVENESS_PESSIMISM":
        e_hat = E_HAT_PESSIMISTIC
    elif condition == "JOINT_MODERATE_MISSPECIFICATION":
        v_hat = multiplicative_lognormal_noise(v, rng, SIGMA_JOINT_V)
        f_hat = multiplicative_lognormal_noise(f, rng, SIGMA_JOINT_F)
        e_hat = E_HAT_JOINT
    else:
        raise ValueError(condition)

    return EstimateBundle(
        value_hat=v_hat,
        fp_hat=f_hat,
        e_hat=e_hat,
        r_hat=r_hat,
        condition=condition,
    )


def _rank_frame(
    df: pd.DataFrame,
    value: np.ndarray,
    fp_cost: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    out["value"] = value
    out["fp_cost"] = fp_cost
    return out


def select_with_quantities(
    df: pd.DataFrame,
    policy: PolicyName,
    capacity: int,
    value: np.ndarray,
    fp_cost: np.ndarray,
    review_cost: float,
    review_effectiveness: float,
) -> np.ndarray:
    rank_df = _rank_frame(df, value, fp_cost)
    return select_for_review(
        rank_df,
        policy,
        capacity,
        review_cost=review_cost,
        review_effectiveness=review_effectiveness,
    )


def evaluate_mask_true(
    df: pd.DataFrame,
    reviewed: np.ndarray,
    e_true: float,
    r_true: float,
) -> dict:
    """Evaluation uses TRUE v/f/e/r only."""
    return total_system_cost(
        df["y"].to_numpy(),
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        reviewed,
        r_true,
        e_true,
    )


def oracle_benefit_scores(
    df: pd.DataFrame,
    e_true: float,
    r_true: float,
) -> np.ndarray:
    return expected_benefit(
        df["p_hat"].to_numpy(),
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        r_true,
        e_true,
    )


def estimated_benefit_scores(
    df: pd.DataFrame,
    est: EstimateBundle,
) -> np.ndarray:
    return expected_benefit(
        df["p_hat"].to_numpy(),
        est.value_hat,
        est.fp_hat,
        est.r_hat,
        est.e_hat,
    )


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    def rankdata(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1, dtype=float)
        sorted_x = x[order]
        i = 0
        while i < len(sorted_x):
            j = i
            while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            if j > i:
                mid = 0.5 * (i + 1 + j + 1)
                ranks[order[i : j + 1]] = mid
            i = j + 1
        return ranks

    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    if denom <= 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def run_policy_row(
    df: pd.DataFrame,
    label: PolicyLabel,
    capacity: int,
    e_true: float,
    r_true: float,
    est: EstimateBundle,
) -> dict:
    """
    Information hierarchy:
      B:        p only
      C_oracle: p + true v          (reference)
      C_est:    p + v_hat           (deployable)
      D1_oracle / D2_oracle: true decision economics (reference)
      D1_est / D2_est: estimated decision economics (deployable)

    C ranking uses e_rank=1 so score = p·v (oracle) or p·v_hat (est).
    Evaluation always uses true v/f/e_true/r_true.
    """
    v = df["value"].to_numpy()
    f = df["fp_cost"].to_numpy()
    abstention_rate = float("nan")
    # C does not use e or f in its score; pin e_rank=1 → p*v
    e_rank_c = 1.0

    if label == "B_topk_probability":
        mask = select_with_quantities(
            df, "B_topk_probability", capacity, v, f, r_true, e_true
        )
        base_policy = "B_topk_probability"
        info_set = "probability_only"
        deployable = True
        is_oracle = False
    elif label == "C_oracle":
        mask = select_with_quantities(
            df, "C_expected_exposure", capacity, v, f, r_true, e_rank_c
        )
        base_policy = "C_expected_exposure"
        info_set = "probability_plus_true_exposure"
        deployable = False
        is_oracle = True
    elif label == "C_est":
        mask = select_with_quantities(
            df,
            "C_expected_exposure",
            capacity,
            est.value_hat,  # same v_hat as D-est under value noise
            f,  # fp unused by C ranking
            r_true,
            e_rank_c,
        )
        base_policy = "C_expected_exposure"
        info_set = "probability_plus_estimated_exposure"
        deployable = True
        is_oracle = False
    elif label == "D1_oracle":
        mask = select_with_quantities(
            df, "D1_cost_aware_topk", capacity, v, f, r_true, e_true
        )
        base_policy = "D1_cost_aware_topk"
        info_set = "full_true_decision_economics"
        deployable = False
        is_oracle = True
    elif label == "D2_oracle":
        mask = select_with_quantities(
            df, "D2_cost_aware_abstention", capacity, v, f, r_true, e_true
        )
        base_policy = "D2_cost_aware_abstention"
        info_set = "full_true_decision_economics_plus_abstention"
        deployable = False
        is_oracle = True
        b = oracle_benefit_scores(df, e_true, r_true)
        abstention_rate = float((b <= 0).mean())
    elif label == "D1_est":
        mask = select_with_quantities(
            df,
            "D1_cost_aware_topk",
            capacity,
            est.value_hat,
            est.fp_hat,
            est.r_hat,
            est.e_hat,
        )
        base_policy = "D1_cost_aware_topk"
        info_set = "full_estimated_decision_economics"
        deployable = True
        is_oracle = False
    elif label == "D2_est":
        mask = select_with_quantities(
            df,
            "D2_cost_aware_abstention",
            capacity,
            est.value_hat,
            est.fp_hat,
            est.r_hat,
            est.e_hat,
        )
        base_policy = "D2_cost_aware_abstention"
        info_set = "full_estimated_decision_economics_plus_abstention"
        deployable = True
        is_oracle = False
        b = estimated_benefit_scores(df, est)
        abstention_rate = float((b <= 0).mean())
    else:
        raise ValueError(label)

    metrics = evaluate_mask_true(df, mask, e_true, r_true)
    d1_oracle_mask = select_with_quantities(
        df, "D1_cost_aware_topk", capacity, v, f, r_true, e_true
    )
    inter = float((mask & d1_oracle_mask).sum())
    union = float((mask | d1_oracle_mask).sum())
    overlap = inter / union if union > 0 else float("nan")

    o_scores = oracle_benefit_scores(df, e_true, r_true)
    if label in ("D1_est", "D2_est"):
        e_scores = estimated_benefit_scores(df, est)
    elif label == "C_est":
        e_scores = df["p_hat"].to_numpy() * est.value_hat
    elif label == "C_oracle":
        e_scores = df["p_hat"].to_numpy() * v
    else:
        e_scores = o_scores
    rho = spearman_corr(o_scores, e_scores)

    metrics.update(
        {
            "policy_label": label,
            "base_policy": base_policy,
            "info_set": info_set,
            "deployable": deployable,
            "is_oracle_reference": is_oracle,
            "condition": est.condition,
            "e_true": e_true,
            "e_hat": est.e_hat,
            "r_true": r_true,
            "r_hat": est.r_hat,
            "abstention_rate": abstention_rate,
            "overlap_with_d1_oracle": round(overlap, 6),
            "spearman_vs_oracle_benefit": round(rho, 6),
            "review_volume": int(mask.sum()),
            "capacity_utilization": round(
                float(mask.sum()) / capacity if capacity else 0.0, 4
            ),
        }
    )
    return metrics


CONDITIONS: tuple[ConditionName, ...] = (
    "ORACLE",
    "VALUE_NOISE_LOW",
    "VALUE_NOISE_HIGH",
    "FP_COST_NOISE_LOW",
    "FP_COST_NOISE_HIGH",
    "REVIEW_EFFECTIVENESS_OPTIMISM",
    "REVIEW_EFFECTIVENESS_PESSIMISM",
    "JOINT_MODERATE_MISSPECIFICATION",
)

POLICY_LABELS: tuple[PolicyLabel, ...] = (
    "B_topk_probability",
    "C_oracle",
    "C_est",
    "D1_oracle",
    "D2_oracle",
    "D1_est",
    "D2_est",
)

# Deployable fair set under misspecification
DEPLOYABLE_LABELS: tuple[PolicyLabel, ...] = (
    "B_topk_probability",
    "C_est",
    "D1_est",
    "D2_est",
)

ORACLE_LABELS: tuple[PolicyLabel, ...] = (
    "C_oracle",
    "D1_oracle",
    "D2_oracle",
)

POLICY_INFO_TABLE = [
    {
        "policy": "B_topk_probability",
        "information_required": "prediction probability p only",
        "uses_estimated_exposure": False,
        "uses_estimated_fp_action_cost": False,
        "can_abstain": False,
        "oracle_or_deployable": "deployable",
    },
    {
        "policy": "C_oracle",
        "information_required": "p + true economic loss exposure v",
        "uses_estimated_exposure": False,
        "uses_estimated_fp_action_cost": False,
        "can_abstain": False,
        "oracle_or_deployable": "oracle_reference",
    },
    {
        "policy": "C_est",
        "information_required": "p + estimated exposure v_hat",
        "uses_estimated_exposure": True,
        "uses_estimated_fp_action_cost": False,
        "can_abstain": False,
        "oracle_or_deployable": "deployable",
    },
    {
        "policy": "D1_oracle",
        "information_required": "p + true v,f,e,r (full decision economics)",
        "uses_estimated_exposure": False,
        "uses_estimated_fp_action_cost": False,
        "can_abstain": False,
        "oracle_or_deployable": "oracle_reference",
    },
    {
        "policy": "D2_oracle",
        "information_required": "same as D1_oracle + abstain if benefit≤0",
        "uses_estimated_exposure": False,
        "uses_estimated_fp_action_cost": False,
        "can_abstain": True,
        "oracle_or_deployable": "oracle_reference",
    },
    {
        "policy": "D1_est",
        "information_required": "p + estimated v_hat,f_hat,e_hat,r_hat",
        "uses_estimated_exposure": True,
        "uses_estimated_fp_action_cost": True,
        "can_abstain": False,
        "oracle_or_deployable": "deployable",
    },
    {
        "policy": "D2_est",
        "information_required": "same as D1_est + abstain if estimated benefit≤0",
        "uses_estimated_exposure": True,
        "uses_estimated_fp_action_cost": True,
        "can_abstain": True,
        "oracle_or_deployable": "deployable",
    },
]

NOISE_DOCUMENTATION = {
    "SIGMA_VALUE_LOW": SIGMA_VALUE_LOW,
    "SIGMA_VALUE_HIGH": SIGMA_VALUE_HIGH,
    "SIGMA_FP_LOW": SIGMA_FP_LOW,
    "SIGMA_FP_HIGH": SIGMA_FP_HIGH,
    "SIGMA_JOINT_V": SIGMA_JOINT_V,
    "SIGMA_JOINT_F": SIGMA_JOINT_F,
    "cv_low_target": 0.20,
    "cv_high_target": 0.50,
    "cv_joint_target": 0.30,
    "mean_one": True,
    "mu_formula": "-sigma^2/2",
    "E_TRUE_DEFAULT": E_TRUE_DEFAULT,
    "E_HAT_OPTIMISTIC": E_HAT_OPTIMISTIC,
    "E_HAT_PESSIMISTIC": E_HAT_PESSIMISTIC,
    "E_HAT_JOINT": E_HAT_JOINT,
    "REVIEW_COST_PER_CASE": REVIEW_COST_PER_CASE,
    "v_definition": (
        "true economic loss exposure conditional on fraud "
        "(not merely transaction value)"
    ),
    "C_score": "p * v (oracle) or p * v_hat (est); e not in C information set",
    "fair_value_noise": "C_est and D*_est share identical v_hat draws",
}
