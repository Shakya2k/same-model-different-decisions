"""
Shared generative model and review policies for Experiments 01–02.

y ~ Bernoulli(p_base). Policies may see temperature-scaled p_hat.
Same y across temperatures for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PolicyName = Literal[
    "A_fixed_threshold_fifo",
    "B_topk_probability",
    "C_expected_exposure",
    "D1_cost_aware_topk",
    "D2_cost_aware_abstention",
]

LABELS = {
    "A_fixed_threshold_fifo": "A Fixed threshold + FIFO",
    "B_topk_probability": "B Top-k probability",
    "C_expected_exposure": "C Expected exposure",
    "D1_cost_aware_topk": "D1 Cost-aware top-K",
    "D2_cost_aware_abstention": "D2 Cost-aware + abstention",
}

NEAR_EQUIVALENCE_PCT = 0.01  # pre-specified in Patch 2; do not change
NEAR_EQUIVALENCE_SENSITIVITY = (0.005, 0.01, 0.02)
REVIEW_COST_PER_CASE = 25.0
FIXED_THRESHOLD = 0.50


@dataclass(frozen=True)
class GenerativeParams:
    n: int = 100_000
    prevalence_intercept: float = -3.9  # alpha
    latent_coef: float = 1.6  # beta
    score_noise_sd: float = 0.35  # sigma_score
    temperature: float = 1.0  # T applied to p_base only
    # value columns = stylized economic loss exposure conditional on fraud (v)
    value_sigma: float = 1.1
    value_mean: float = 4.2
    constant_value: float | None = None
    fp_mean: float = 40.0
    fp_sigma: float = 0.4
    constant_fp: float | None = None
    review_cost: float = REVIEW_COST_PER_CASE
    # e: fraction of fraud loss prevented if reviewed (deterministic, not Bernoulli)
    review_effectiveness: float = 1.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def generate_base_dataset(seed: int, params: GenerativeParams) -> pd.DataFrame:
    """Draw y from p_base once. Apply temperature separately for policy scores."""
    rng = np.random.default_rng(seed)
    n = params.n
    z = rng.normal(size=n)
    eps = rng.normal(0, params.score_noise_sd, size=n)
    s = params.prevalence_intercept + params.latent_coef * z + eps
    p_base = _sigmoid(s)
    y = rng.binomial(1, p_base)

    if params.constant_value is not None:
        value = np.full(n, float(params.constant_value))
    elif params.value_sigma <= 0:
        value = np.full(n, float(np.exp(params.value_mean)))
    else:
        value = np.clip(
            rng.lognormal(mean=params.value_mean, sigma=params.value_sigma, size=n),
            5,
            50_000,
        )

    if params.constant_fp is not None:
        fp_cost = np.full(n, float(params.constant_fp))
    elif params.fp_sigma <= 0:
        fp_cost = np.full(n, float(params.fp_mean))
    else:
        fp_cost = np.clip(
            rng.lognormal(mean=np.log(params.fp_mean), sigma=params.fp_sigma, size=n),
            5,
            2_000,
        )

    arrival_index = rng.permutation(n)
    return pd.DataFrame(
        {
            "y": y.astype(int),
            "p_base": p_base,
            "p_hat": p_base.copy(),  # default T=1
            "value": value,
            "fp_cost": fp_cost,
            "arrival_index": arrival_index,
        }
    )


def apply_temperature(df: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """Return copy with p_hat = sigmoid(logit(p_base)/T). y unchanged."""
    out = df.copy()
    t = max(float(temperature), 1e-8)
    out["p_hat"] = _sigmoid(_logit(out["p_base"].to_numpy()) / t)
    return out


def generate_transactions(seed: int, params: GenerativeParams) -> pd.DataFrame:
    """Convenience: base dataset + temperature for params.temperature."""
    base = generate_base_dataset(seed, params)
    return apply_temperature(base, params.temperature)


def expected_benefit(
    p: np.ndarray,
    value: np.ndarray,
    fp_cost: np.ndarray,
    review_cost: float,
    review_effectiveness: float = 1.0,
) -> np.ndarray:
    """Expected reduction in modeled system cost from reviewing one case."""
    e = float(review_effectiveness)
    return p * e * value - review_cost - (1.0 - p) * fp_cost


def select_for_review(
    df: pd.DataFrame,
    policy: PolicyName,
    capacity: int,
    review_cost: float = REVIEW_COST_PER_CASE,
    threshold: float = FIXED_THRESHOLD,
    review_effectiveness: float = 1.0,
) -> np.ndarray:
    """Select up to `capacity` cases. Never reads y."""
    cols = [c for c in df.columns if c != "y"]
    work = df[cols]
    assert "y" not in work.columns

    p = work["p_hat"].to_numpy()
    value = work["value"].to_numpy()
    fp_cost = work["fp_cost"].to_numpy()
    arrival = work["arrival_index"].to_numpy()
    n = len(work)
    mask = np.zeros(n, dtype=bool)
    benefit = expected_benefit(p, value, fp_cost, review_cost, review_effectiveness)

    if policy == "A_fixed_threshold_fifo":
        eligible = np.flatnonzero(p > threshold)
        order = eligible[np.argsort(arrival[eligible], kind="mergesort")]
        mask[order[:capacity]] = True
        return mask

    if policy == "B_topk_probability":
        order = np.argsort(-p, kind="mergesort")[:capacity]
        mask[order] = True
        return mask

    if policy == "C_expected_exposure":
        # Include e so C scales with D when effectiveness changes
        score = p * review_effectiveness * value
        order = np.argsort(-score, kind="mergesort")[:capacity]
        mask[order] = True
        return mask

    if policy == "D1_cost_aware_topk":
        # Capacity is a hard ceiling; fills K even if marginal benefit is negative
        order = np.argsort(-benefit, kind="mergesort")[:capacity]
        mask[order] = True
        return mask

    if policy == "D2_cost_aware_abstention":
        positive = np.flatnonzero(benefit > 0)
        order = positive[np.argsort(-benefit[positive], kind="mergesort")]
        mask[order[:capacity]] = True
        return mask

    raise ValueError(policy)


def total_system_cost(
    y: np.ndarray,
    value: np.ndarray,
    fp_cost: np.ndarray,
    reviewed: np.ndarray,
    review_cost: float,
    review_effectiveness: float = 1.0,
) -> dict:
    """
    Realized simulated system cost.

    Unreviewed fraud: (1-q)*y*v
    Reviewed residual fraud: q*y*(1-e)*v   (e = fraction prevented if reviewed)
    Review cost: q*r
    FP friction: q*(1-y)*f
    """
    q = reviewed.astype(float)
    yf = y.astype(float)
    e = float(review_effectiveness)
    fraud_loss_remaining = float(
        ((1 - q) * yf * value + q * yf * (1 - e) * value).sum()
    )
    fraud_dollars_captured = float((q * yf * e * value).sum())
    review_cost_total = float((q * review_cost).sum())
    fp_intervention_cost = float((q * (1 - yf) * fp_cost).sum())
    review_volume = int(reviewed.sum())
    tp = int((reviewed & (y == 1)).sum())
    precision = tp / review_volume if review_volume else 0.0
    return {
        "review_volume": review_volume,
        "precision_among_reviewed": round(precision, 4),
        "fraud_dollars_captured": round(fraud_dollars_captured, 2),
        "fraud_loss_remaining": round(fraud_loss_remaining, 2),
        "fp_intervention_cost": round(fp_intervention_cost, 2),
        "review_cost": round(review_cost_total, 2),
        "total_system_cost": round(
            fraud_loss_remaining + fp_intervention_cost + review_cost_total, 2
        ),
        "review_effectiveness": e,
    }


def evaluate_policy(
    df: pd.DataFrame,
    policy: PolicyName,
    capacity: int,
    review_cost: float = REVIEW_COST_PER_CASE,
    review_effectiveness: float = 1.0,
) -> dict:
    mask = select_for_review(
        df,
        policy,
        capacity,
        review_cost=review_cost,
        review_effectiveness=review_effectiveness,
    )
    assert int(mask.sum()) <= capacity
    metrics = total_system_cost(
        df["y"].to_numpy(),
        df["value"].to_numpy(),
        df["fp_cost"].to_numpy(),
        mask,
        review_cost,
        review_effectiveness,
    )
    metrics["policy"] = policy
    metrics["same_predictions"] = "Yes"
    metrics["capacity_utilization"] = round(
        metrics["review_volume"] / capacity if capacity else 0.0, 4
    )
    return metrics


def predictive_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """ROC-AUC (Mann–Whitney), Brier, log loss."""
    y = y.astype(float)
    p = np.clip(p.astype(float), 1e-12, 1 - 1e-12)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        order = np.argsort(p)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(p) + 1, dtype=float)
        sorted_p = p[order]
        i = 0
        while i < len(sorted_p):
            j = i
            while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
                j += 1
            if j > i:
                mid = 0.5 * (i + 1 + j + 1)
                ranks[order[i : j + 1]] = mid
            i = j + 1
        auc = (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (
            len(pos) * len(neg)
        )
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {"roc_auc": float(auc), "brier": brier, "log_loss": log_loss}


def policies_use_only_predictions(df: pd.DataFrame, capacity: int = 100) -> None:
    df2 = df.copy()
    df2["y"] = 1 - df2["y"]
    for policy in LABELS:
        m1 = select_for_review(df, policy, capacity)  # type: ignore[arg-type]
        m2 = select_for_review(df2, policy, capacity)  # type: ignore[arg-type]
        assert np.array_equal(m1, m2), f"Leakage in {policy}"
