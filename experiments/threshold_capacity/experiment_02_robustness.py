#!/usr/bin/env python3
"""Experiment 02 — Patch 3 multi-seed robustness (10 seeds × 135 cells)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decision_model import (
    LABELS,
    NEAR_EQUIVALENCE_PCT,
    NEAR_EQUIVALENCE_SENSITIVITY,
    GenerativeParams,
    apply_temperature,
    evaluate_policy,
    generate_base_dataset,
)

N_SEEDS = 10
BASE_SEED = 42
CAPACITY_GRID = [500, 1_000, 2_000, 5_000, 10_000]
PREVALENCE = {"low": -4.6, "baseline": -3.9, "high": -3.2}
TEMPERATURES = {
    "calibrated_by_construction_T1.0": 1.0,
    "temperature_T1.5": 1.5,
    "temperature_T2.5": 2.5,
}
FP_COST = {"low": 15.0, "baseline": 40.0, "high": 120.0}

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
REPO_FIG = ROOT.parents[1] / "figures"
POLICIES = list(LABELS.keys())


def main() -> None:
    for d in (FIG, OUT, REPO_FIG):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    cell_id = 0
    seeds = [BASE_SEED + 1000 + i * 17 for i in range(N_SEEDS)]
    scenarios = 0
    for prev, alpha in PREVALENCE.items():
        for cal_label, t in TEMPERATURES.items():
            for fp_k, fp_mean in FP_COST.items():
                for cap in CAPACITY_GRID:
                    scenarios += 1
                    cell_id += 1
                    for seed0 in seeds:
                        seed = seed0 + cell_id * 9973
                        params = GenerativeParams(
                            prevalence_intercept=alpha, fp_mean=fp_mean
                        )
                        base = generate_base_dataset(seed, params)
                        df = apply_temperature(base, t)
                        for policy in POLICIES:
                            r = evaluate_policy(df, policy, cap)  # type: ignore[arg-type]
                            r.update(
                                {
                                    "prevalence": prev,
                                    "calibration": cal_label,
                                    "temperature": t,
                                    "fp_cost_regime": fp_k,
                                    "capacity": cap,
                                    "seed": seed,
                                }
                            )
                            rows.append(r)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "experiment_02_full_multiseed.csv", index=False)

    keys = ["prevalence", "calibration", "fp_cost_regime", "capacity", "policy"]
    # Uncertainty: normal approx mean ± 1.96*sd/sqrt(n) (Monte Carlo / large-n CLT interval)
    agg = (
        results.groupby(keys)["total_system_cost"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_cost", "std": "std_cost", "count": "n_seeds"})
    )
    agg["ci95_low"] = agg["mean_cost"] - 1.96 * agg["std_cost"] / np.sqrt(agg["n_seeds"])
    agg["ci95_high"] = agg["mean_cost"] + 1.96 * agg["std_cost"] / np.sqrt(agg["n_seeds"])
    agg["interval_method"] = "normal_approx_mean_pm_1.96_se"
    agg.to_csv(OUT / "experiment_02_policy_means.csv", index=False)

    scenario_keys = ["prevalence", "calibration", "fp_cost_regime", "capacity"]
    idx = agg.groupby(scenario_keys)["mean_cost"].idxmin()
    best = agg.loc[idx].rename(
        columns={"policy": "best_policy", "mean_cost": "best_mean_cost"}
    )
    merged = agg.merge(
        best[scenario_keys + ["best_policy", "best_mean_cost"]], on=scenario_keys
    )
    merged["abs_diff_from_best"] = merged["mean_cost"] - merged["best_mean_cost"]
    merged["pct_diff_from_best"] = merged["abs_diff_from_best"] / merged["best_mean_cost"]
    merged["near_eq_1pct"] = merged["pct_diff_from_best"] <= NEAR_EQUIVALENCE_PCT + 1e-12
    for band in NEAR_EQUIVALENCE_SENSITIVITY:
        merged[f"near_eq_{band}"] = merged["pct_diff_from_best"] <= band + 1e-12
    merged.to_csv(OUT / "experiment_02_effect_sizes.csv", index=False)

    headlines = []
    a_win_rows = []
    for _, g in merged.groupby(scenario_keys):
        best_row = g.loc[g["mean_cost"].idxmin()]
        near = g[g["near_eq_1pct"]]["policy"].tolist()
        scen = results[
            (results["prevalence"] == best_row["prevalence"])
            & (results["calibration"] == best_row["calibration"])
            & (results["fp_cost_regime"] == best_row["fp_cost_regime"])
            & (results["capacity"] == best_row["capacity"])
        ]
        seed_best = scen.loc[scen.groupby("seed")["total_system_cost"].idxmin()]
        win_rate = float((seed_best["policy"] == best_row["policy"]).mean())
        status = "clear_winner"
        if len(near) > 1:
            status = "near_equivalent"
        if win_rate < 0.7:
            status = (
                "unstable_across_seeds"
                if status != "near_equivalent"
                else "near_equivalent_unstable"
            )
        headlines.append(
            {
                **{k: best_row[k] for k in scenario_keys},
                "best_policy": best_row["policy"],
                "best_mean_cost": best_row["mean_cost"],
                "near_equivalent_policies": ",".join(near),
                "seed_win_rate_of_best": round(win_rate, 3),
                "status": status,
            }
        )
        if best_row["policy"] == "A_fixed_threshold_fifo":
            # diagnostic volumes/costs at mean across seeds for this cell
            cell = (
                scen.groupby("policy")[
                    [
                        "review_volume",
                        "fraud_loss_remaining",
                        "fp_intervention_cost",
                        "review_cost",
                        "total_system_cost",
                        "precision_among_reviewed",
                    ]
                ]
                .mean()
                .reset_index()
            )
            for _, crow in cell.iterrows():
                a_win_rows.append(
                    {
                        **{k: best_row[k] for k in scenario_keys},
                        "policy": crow["policy"],
                        "mean_review_volume": crow["review_volume"],
                        "mean_fraud_loss": crow["fraud_loss_remaining"],
                        "mean_fp_cost": crow["fp_intervention_cost"],
                        "mean_review_cost": crow["review_cost"],
                        "mean_total_cost": crow["total_system_cost"],
                        "mean_precision": crow["precision_among_reviewed"],
                    }
                )

    head_df = pd.DataFrame(headlines)
    head_df.to_csv(OUT / "experiment_02_scenario_headlines.csv", index=False)
    pd.DataFrame(a_win_rows).to_csv(OUT / "experiment_02_a_win_diagnostic.csv", index=False)
    win_counts = head_df["best_policy"].value_counts()
    status_counts = head_df["status"].value_counts()
    win_counts.to_csv(OUT / "experiment_02_win_frequency.csv")

    # Heatmap with NE markers
    heat = head_df[
        (head_df["prevalence"] == "baseline")
        & (head_df["calibration"] == "calibrated_by_construction_T1.0")
    ]
    pivot = heat.pivot(
        index="capacity", columns="fp_cost_regime", values="best_policy"
    ).reindex(index=CAPACITY_GRID, columns=["low", "baseline", "high"])
    status_pivot = heat.pivot(
        index="capacity", columns="fp_cost_regime", values="status"
    ).reindex(index=CAPACITY_GRID, columns=["low", "baseline", "high"])
    code = {p: i for i, p in enumerate(POLICIES)}
    pivot_num = pivot.map(lambda p: code.get(p, -1)).astype(float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(pivot_num.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low", "baseline", "high"])
    ax.set_yticks(range(len(CAPACITY_GRID)))
    ax.set_yticklabels(CAPACITY_GRID)
    ax.set_xlabel("FP cost regime")
    ax.set_ylabel("Capacity")
    ax.set_title("Dominant policy (baseline, T=1); NE = near-equivalent")
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(POLICIES)))
    cbar.ax.set_yticklabels([LABELS[p].split()[0] for p in POLICIES])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            pol = str(pivot.iloc[i, j])
            st = str(status_pivot.iloc[i, j])
            label = pol.split("_")[0]
            if "near" in st:
                label += "·NE"
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    for dest in (FIG, OUT, REPO_FIG):
        fig.savefig(dest / "best_policy_heatmap.png", dpi=160)
    plt.close(fig)

    # A-win diagnostic markdown
    a_diag = pd.DataFrame(a_win_rows)
    n_a = int((head_df["best_policy"] == "A_fixed_threshold_fifo").sum())
    a_md = f"""# A-win diagnostic (Patch 3)

A (fixed threshold + FIFO) was primary best-by-mean in **{n_a}** / {len(head_df)} scenarios.

## Mechanism summary

Compare mean review volumes: A typically reviews **fewer** cases than forced top-K policies when the mass above p>0.5 is limited, lowering review cost and FP friction while accepting higher residual fraud loss. Wins are concentrated where friction/review costs outweigh residual loss under the stylized objective — not treated as implementation errors.

See `outputs/experiment_02_a_win_diagnostic.csv`.
"""
    if len(a_diag):
        # summarize volume ratios
        a_only = a_diag[a_diag["policy"] == "A_fixed_threshold_fifo"]
        b_only = a_diag[a_diag["policy"] == "B_topk_probability"]
        if len(a_only) and len(b_only):
            a_md += f"\nMean A volume (A-win cells): {a_only['mean_review_volume'].mean():.1f}\n"
            a_md += f"Mean B volume (same cells): {b_only['mean_review_volume'].mean():.1f}\n"
    (ROOT / "a_win_diagnostic.md").write_text(a_md)

    md = f"""# Experiment 02 Robustness (Patch 3)

## Framing

Policy sensitivity under fixed predictions — not universal cost-aware dominance.

## Design

- Scenarios: {scenarios}
- Seeds/scenario: {N_SEEDS}
- Dataset draws: {scenarios * N_SEEDS}
- Policy evaluations: {len(results)}
- Interval method: normal approx mean ± 1.96·SE (Monte Carlo CLT), n={N_SEEDS} seeds
- Near-equivalence: {NEAR_EQUIVALENCE_PCT:.0%} (pre-specified); sensitivity {NEAR_EQUIVALENCE_SENSITIVITY}
- Generative: y from p_base (calibrated by construction); p(T)=sigmoid(logit(p_base)/T)

## Primary best counts

{win_counts.to_string()}

## Status counts

{status_counts.to_string()}

Integrity notices apply (synthetic; stylized objective).
"""
    (ROOT / "robustness_results.md").write_text(md)
    print(win_counts)
    print(status_counts)
    print(f"scenarios={scenarios} seeds={N_SEEDS} rows={len(results)}")


if __name__ == "__main__":
    main()
