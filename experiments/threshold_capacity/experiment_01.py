#!/usr/bin/env python3
"""Experiment 01 — Patch 3 canonical seed-42 (illustrative)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from decision_model import (
    LABELS,
    GenerativeParams,
    apply_temperature,
    evaluate_policy,
    generate_base_dataset,
    policies_use_only_predictions,
    predictive_metrics,
)

SEED = 42
CAPACITY = 2_000
ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
REPO_FIG = ROOT.parents[1] / "figures"
POLICIES = list(LABELS.keys())


def main() -> None:
    for d in (FIG, OUT, REPO_FIG):
        d.mkdir(parents=True, exist_ok=True)

    params = GenerativeParams(temperature=1.0)
    base = generate_base_dataset(SEED, params)
    df = apply_temperature(base, 1.0)
    policies_use_only_predictions(df, CAPACITY)

    rows = [evaluate_policy(df, p, CAPACITY) for p in POLICIES]  # type: ignore[arg-type]
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "experiment_01_summary.csv", index=False)

    # Attribution
    costs = {r["policy"]: r["total_system_cost"] for r in rows}
    attr = pd.DataFrame(
        [
            {
                "metric": "ranking_gain_B_to_D1",
                "value": costs["B_topk_probability"] - costs["D1_cost_aware_topk"],
            },
            {
                "metric": "ranking_gain_C_to_D1",
                "value": costs["C_expected_exposure"] - costs["D1_cost_aware_topk"],
            },
            {
                "metric": "abstention_gain_D1_to_D2",
                "value": costs["D1_cost_aware_topk"] - costs["D2_cost_aware_abstention"],
            },
        ]
    )
    attr.to_csv(OUT / "experiment_01_attribution.csv", index=False)

    # Predictive metrics by T (same y)
    metric_rows = []
    for t in (1.0, 1.5, 2.5):
        dft = apply_temperature(base, t)
        m = predictive_metrics(dft["y"].to_numpy(), dft["p_hat"].to_numpy())
        m["T"] = t
        m["label"] = (
            "calibrated_by_construction"
            if t == 1.0
            else f"temperature_T{t}"
        )
        metric_rows.append(m)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUT / "experiment_01_predictive_metrics_by_T.csv", index=False)

    # Figures
    plot_df = results.copy()
    plot_df["label"] = plot_df["policy"].map(LABELS)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["label"], plot_df["total_system_cost"], color="#2f6fed")
    ax.set_ylabel("Total system cost")
    ax.set_title(
        "Total System Cost by Decision Policy\n"
        "Illustrative canonical simulation — seed 42 (Patch 3)"
    )
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    for dest in (FIG, OUT, REPO_FIG):
        fig.savefig(dest / "total_system_cost_by_policy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(metrics_df["T"], metrics_df["roc_auc"], marker="o", label="ROC-AUC")
    ax.plot(metrics_df["T"], metrics_df["brier"], marker="s", label="Brier")
    ax.plot(metrics_df["T"], metrics_df["log_loss"], marker="^", label="Log loss")
    ax.set_xlabel("Temperature T")
    ax.set_title("Predictive metrics vs calibration temperature (same y)")
    ax.legend()
    fig.tight_layout()
    for dest in (FIG, OUT, REPO_FIG):
        fig.savefig(dest / "predictive_metrics_vs_T.png", dpi=160)
    plt.close(fig)

    # Decision cost vs T for B,C,D1,D2
    cost_t_rows = []
    for t in (1.0, 1.5, 2.5):
        dft = apply_temperature(base, t)
        for p in [
            "B_topk_probability",
            "C_expected_exposure",
            "D1_cost_aware_topk",
            "D2_cost_aware_abstention",
        ]:
            r = evaluate_policy(dft, p, CAPACITY)  # type: ignore[arg-type]
            cost_t_rows.append({"T": t, "policy": p, "total_system_cost": r["total_system_cost"]})
    cost_t = pd.DataFrame(cost_t_rows)
    cost_t.to_csv(OUT / "experiment_01_cost_vs_T.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, g in cost_t.groupby("policy"):
        ax.plot(g["T"], g["total_system_cost"], marker="o", label=LABELS[p])
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Total system cost")
    ax.set_title("Decision cost vs calibration temperature (seed 42)")
    ax.legend()
    fig.tight_layout()
    for dest in (FIG, OUT, REPO_FIG):
        fig.savefig(dest / "decision_cost_vs_T.png", dpi=160)
    plt.close(fig)

    # Attribution bars
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(attr["metric"], attr["value"], color="#34d399")
    ax.set_ylabel("Cost reduction (positive = improvement)")
    ax.set_title("Ranking gain vs abstention gain (seed 42)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    for dest in (FIG, OUT, REPO_FIG):
        fig.savefig(dest / "ranking_vs_abstention_gain.png", dpi=160)
    plt.close(fig)

    md = f"""# Experiment 01 Results (Patch 3)

## Framing

Holding model predictions fixed does not hold operational outcomes fixed. Downstream
policy choices — ranking, action costs, capacity, calibration, and abstention —
can materially change system performance.

## Integrity

This is a synthetic experiment, not evidence of realized fraud reduction in a production financial-services environment. Results depend on the simulated fraud prevalence, probability calibration, transaction-value distribution, intervention costs and capacity assumptions.

The cost-aware policies are evaluated against the same stylized cost structure used to define expected action value. Advantages reflect this decision problem and are not universal dominance.

## Generative model (Patch 3)

y ~ Bernoulli(p_base) with p_base = sigmoid(α + βz + ε) — **calibrated by construction**.
Policies use p(T)=sigmoid(logit(p_base)/T). Same y across T.

## Policies

A FIFO threshold · B top-K p · C top-K p·e·v · D1 cost-aware top-K · D2 cost-aware + abstention

## Canonical table (seed {SEED}, T=1, e=1, K={CAPACITY})

| Policy | Volume | Precision | Total cost |
| --- | ---: | ---: | ---: |
"""
    for r in rows:
        md += f"| {LABELS[r['policy']]} | {r['review_volume']} | {r['precision_among_reviewed']:.4f} | {r['total_system_cost']:,.2f} |\n"
    md += f"""
## Attribution

{attr.to_string(index=False)}

## Predictive metrics by T

{metrics_df.to_string(index=False)}

Patch-2 baselines preserved under `baselines/patch2_seed42_design/` (superseded for calibration).
"""
    (ROOT / "results.md").write_text(md)
    print(results[["policy", "review_volume", "total_system_cost"]].to_string(index=False))
    print(attr.to_string(index=False))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
