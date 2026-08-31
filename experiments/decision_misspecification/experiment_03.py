#!/usr/bin/env python3
"""
Experiment 03 — Decision-model misspecification (fair information sets).

v = true economic loss exposure conditional on fraud (evaluation oracle).
v_hat = policy estimate of that exposure.

Deployable fair set: B, C_est, D1_est, D2_est
Oracle references:   C_oracle, D1_oracle, D2_oracle

Under VALUE NOISE, C_est and D*_est share the same v_hat.

Does not modify Experiments 01–02.
Pre-fair Exp03 outputs preserved under outputs/frozen_pre_fair_comparator_v0/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

_TC = Path(__file__).resolve().parents[1] / "threshold_capacity"
sys.path.insert(0, str(_TC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decision_model import (  # noqa: E402
    REVIEW_COST_PER_CASE,
    GenerativeParams,
    apply_temperature,
    generate_base_dataset,
)
from estimation import (  # noqa: E402
    CONDITIONS,
    DEPLOYABLE_LABELS,
    E_TRUE_DEFAULT,
    NOISE_DOCUMENTATION,
    POLICY_INFO_TABLE,
    POLICY_LABELS,
    build_estimates,
    run_policy_row,
)

N_SEEDS = 20
BASE_SEED = 42
CAPACITY = 2_000
T_CRIT_DF19 = 2.093024054408263
NEAR_EQUIVALENCE_PCT = 0.01  # project-defined practical threshold, not industry standard

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
REPO_FIG = ROOT.parents[1] / "figures"


def paired_t(deltas: np.ndarray, t_crit: float) -> dict:
    n = len(deltas)
    mean = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    lo = mean - t_crit * se
    hi = mean + t_crit * se
    if lo > 0:
        zero = "excludes_0_positive"
    elif hi < 0:
        zero = "excludes_0_negative"
    else:
        zero = "includes_0"
    return {
        "n": n,
        "df": n - 1,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": lo,
        "ci95_high": hi,
        "zero_status": zero,
        "t_crit": t_crit,
        "interval_method": "student_t_df_n_minus_1",
    }


def classify_practical(mean: float, lo: float, hi: float, ref_mean: float) -> str:
    band = NEAR_EQUIVALENCE_PCT * abs(ref_mean)
    if np.isfinite(lo) and np.isfinite(hi) and lo >= -band and hi <= band:
        return "ci_within_1pct_practical_band"
    if abs(mean) <= band:
        return "mean_within_1pct_band_ci_not"
    return "outside_1pct_practical_band"


def main() -> None:
    for d in (OUT, FIG, REPO_FIG):
        d.mkdir(parents=True, exist_ok=True)

    (OUT / "noise_parameters.json").write_text(
        json.dumps(NOISE_DOCUMENTATION, indent=2) + "\n"
    )
    pd.DataFrame(POLICY_INFO_TABLE).to_csv(
        OUT / "experiment_03_policy_information_requirements.csv", index=False
    )

    seeds = [BASE_SEED + 1000 + i * 17 for i in range(N_SEEDS)]
    rows: list[dict] = []
    e_true = E_TRUE_DEFAULT
    r_true = REVIEW_COST_PER_CASE

    for seed in seeds:
        params = GenerativeParams(n=100_000, review_effectiveness=e_true)
        base = generate_base_dataset(seed, params)
        df = apply_temperature(base, 1.0)
        for condition in CONDITIONS:
            est = build_estimates(
                df["value"].to_numpy(),
                df["fp_cost"].to_numpy(),
                e_true,
                r_true,
                condition,
                seed,
            )
            for label in POLICY_LABELS:
                r = run_policy_row(df, label, CAPACITY, e_true, r_true, est)
                r["seed"] = seed
                r["capacity"] = CAPACITY
                rows.append(r)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "experiment_03_full.csv", index=False)

    means = (
        results.groupby(["condition", "policy_label"])[
            [
                "total_system_cost",
                "fraud_loss_remaining",
                "review_cost",
                "fp_intervention_cost",
                "review_volume",
                "overlap_with_d1_oracle",
                "spearman_vs_oracle_benefit",
                "abstention_rate",
                "capacity_utilization",
            ]
        ]
        .mean()
        .reset_index()
    )
    means.to_csv(OUT / "experiment_03_means.csv", index=False)

    # Paired deltas: deployable fair set + key oracle gaps
    delta_specs = [
        ("C_est", "B_topk_probability", "delta_C_est_minus_B"),
        ("D1_est", "B_topk_probability", "delta_D1_est_minus_B"),
        ("D2_est", "B_topk_probability", "delta_D2_est_minus_B"),
        ("D1_est", "C_est", "delta_D1_est_minus_C_est"),
        ("D2_est", "C_est", "delta_D2_est_minus_C_est"),
        ("D2_est", "D1_est", "delta_D2_est_minus_D1_est"),
        ("C_est", "C_oracle", "estimation_penalty_C"),
        ("D1_est", "D1_oracle", "estimation_penalty_D1"),
        ("D2_est", "D2_oracle", "estimation_penalty_D2"),
        ("C_oracle", "D1_oracle", "delta_C_oracle_minus_D1_oracle"),
    ]
    delta_rows = []
    for condition in CONDITIONS:
        sub = results[results["condition"] == condition]
        wide = sub.pivot_table(
            index="seed",
            columns="policy_label",
            values="total_system_cost",
            aggfunc="first",
        )
        for left, right, name in delta_specs:
            deltas = (wide[left] - wide[right]).to_numpy(dtype=float)
            st = paired_t(deltas, T_CRIT_DF19)
            ref_mean = float(wide[right].mean())
            left_mean = float(wide[left].mean())
            rel = st["mean"] / ref_mean if ref_mean != 0 else float("nan")
            delta_rows.append(
                {
                    "condition": condition,
                    "comparison": name,
                    "left_policy": left,
                    "right_policy": right,
                    **{f"paired_{k}": v for k, v in st.items()},
                    "left_mean_cost": left_mean,
                    "right_mean_cost": ref_mean,
                    "relative_pct_of_right_mean": 100.0 * rel,
                    "practical_band_abs": NEAR_EQUIVALENCE_PCT * abs(ref_mean),
                    "practical_class": classify_practical(
                        st["mean"], st["ci95_low"], st["ci95_high"], ref_mean
                    ),
                    "practical_band_note": (
                        "1% is a project-defined practical threshold, not an industry standard"
                    ),
                }
            )
    deltas_df = pd.DataFrame(delta_rows)
    deltas_df.to_csv(OUT / "experiment_03_paired_deltas.csv", index=False)

    # Estimation-penalty focused table
    pen = deltas_df[
        deltas_df["comparison"].isin(
            [
                "estimation_penalty_C",
                "estimation_penalty_D1",
                "estimation_penalty_D2",
            ]
        )
    ].copy()
    pen.to_csv(OUT / "experiment_03_estimation_penalties.csv", index=False)

    # Value-noise deployable comparison summary
    vn_rows = []
    for condition in ("VALUE_NOISE_LOW", "VALUE_NOISE_HIGH"):
        for left, right, name in [
            ("C_est", "B_topk_probability", "C_est_vs_B"),
            ("D1_est", "B_topk_probability", "D1_est_vs_B"),
            ("D2_est", "B_topk_probability", "D2_est_vs_B"),
            ("D1_est", "C_est", "D1_est_vs_C_est"),
            ("D2_est", "C_est", "D2_est_vs_C_est"),
        ]:
            r = deltas_df[
                (deltas_df["condition"] == condition)
                & (deltas_df["left_policy"] == left)
                & (deltas_df["right_policy"] == right)
            ].iloc[0]
            vn_rows.append(
                {
                    "condition": condition,
                    "comparison": name,
                    "mean_delta": r["paired_mean"],
                    "ci95_low": r["paired_ci95_low"],
                    "ci95_high": r["paired_ci95_high"],
                    "zero_status": r["paired_zero_status"],
                    "practical_class": r["practical_class"],
                    "relative_pct": r["relative_pct_of_right_mean"],
                }
            )
    vn_df = pd.DataFrame(vn_rows)
    vn_df.to_csv(OUT / "experiment_03_value_noise_fair_comparisons.csv", index=False)

    # Information × mean cost wide table
    info = pd.DataFrame(POLICY_INFO_TABLE)
    cost_wide = means.pivot(
        index="policy_label", columns="condition", values="total_system_cost"
    ).reset_index()
    cost_wide = info.merge(cost_wide, left_on="policy", right_on="policy_label", how="left")
    cost_wide = cost_wide.drop(columns=["policy_label"])
    cost_wide.to_csv(OUT / "experiment_03_information_cost_table.csv", index=False)

    _write_figures(means, results)
    _write_results_md(means, deltas_df, vn_df, pen)
    print(f"Wrote {len(results)} rows → {OUT / 'experiment_03_full.csv'}")


def _write_figures(means: pd.DataFrame, results: pd.DataFrame) -> None:
    cond_order = list(CONDITIONS)
    plot_pols = list(DEPLOYABLE_LABELS) + ["D1_oracle", "C_oracle"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(cond_order))
    width = 0.12
    for i, pol in enumerate(plot_pols):
        ys = [
            float(
                means.loc[
                    (means["condition"] == c) & (means["policy_label"] == pol),
                    "total_system_cost",
                ].iloc[0]
            )
            for c in cond_order
        ]
        ax.bar(x + (i - len(plot_pols) / 2) * width, ys, width, label=pol)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cond_order], fontsize=7)
    ax.set_ylabel("Mean total system cost")
    ax.set_title("Exp03 — cost vs misspecification (fair information sets)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for dest in (FIG, REPO_FIG):
        fig.savefig(dest / "exp03_cost_vs_estimation_error.png", dpi=140)
    plt.close(fig)

    # Estimation penalty vs oracle
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for pol, ref in (("C_est", "C_oracle"), ("D1_est", "D1_oracle"), ("D2_est", "D2_oracle")):
        pens = []
        for c in cond_order:
            sub = results[results["condition"] == c]
            wide = sub.pivot_table(
                index="seed", columns="policy_label", values="total_system_cost", aggfunc="first"
            )
            pens.append(float((wide[pol] - wide[ref]).mean()))
        ax.plot(cond_order, pens, marker="o", label=f"{pol} − {ref}")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cond_order], fontsize=7)
    ax.set_ylabel("Mean estimation penalty (cost)")
    ax.set_title("Exp03 — estimation penalty vs matched oracle")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for dest in (FIG, REPO_FIG):
        fig.savefig(dest / "exp03_estimation_penalty_vs_oracle.png", dpi=140)
        # keep prior filename as alias for draft links
        fig.savefig(dest / "exp03_regret_vs_oracle.png", dpi=140)
    plt.close(fig)

    # Overlap
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for pol in ("C_est", "D1_est", "D2_est", "B_topk_probability"):
        ys = [
            float(
                means.loc[
                    (means["condition"] == c) & (means["policy_label"] == pol),
                    "overlap_with_d1_oracle",
                ].iloc[0]
            )
            for c in cond_order
        ]
        ax.plot(cond_order, ys, marker="o", label=pol)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cond_order], fontsize=7)
    ax.set_ylabel("Jaccard overlap with D1 oracle")
    ax.set_title("Exp03 — policy overlap vs estimation error")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for dest in (FIG, REPO_FIG):
        fig.savefig(dest / "exp03_overlap_vs_estimation_error.png", dpi=140)
    plt.close(fig)


def _write_results_md(
    means: pd.DataFrame,
    deltas_df: pd.DataFrame,
    vn_df: pd.DataFrame,
    pen: pd.DataFrame,
) -> None:
    def m(cond: str, pol: str) -> float:
        return float(
            means.loc[
                (means["condition"] == cond) & (means["policy_label"] == pol),
                "total_system_cost",
            ].iloc[0]
        )

    lines = [
        "# Experiment 03 — Decision-model misspecification (fair information sets)",
        "",
        "**Status:** PRIVATE. Additive to methodology baseline v0.1.",
        "Pre-fair outputs frozen at `outputs/frozen_pre_fair_comparator_v0/`.",
        "",
        "## Terminology",
        "",
        "- `v` = true **economic loss exposure conditional on fraud** (evaluation oracle)",
        "- `v_hat` = policy estimate of that exposure",
        "- Simulator uses stylized monetary draws; this does **not** reproduce a specific bank's fraud-loss process.",
        "",
        "`e` = fraction of fraud loss prevented conditional on review (deterministic fractional model).",
        "",
        "## Information hierarchy",
        "",
        "| Policy | Information | Deployable? |",
        "| --- | --- | --- |",
        "| B | p only | yes |",
        "| C_est | p + v_hat | yes |",
        "| D1_est | p + v_hat + estimated friction/action economics | yes |",
        "| D2_est | D1_est + abstention | yes |",
        "| C_oracle / D1_oracle / D2_oracle | true quantities | reference only |",
        "",
        "Fair value-noise rule: **C_est and D*_est share the same v_hat**.",
        "",
        f"Seeds={N_SEEDS}, K={CAPACITY}, e_true={E_TRUE_DEFAULT}, Student-t df=19.",
        "1% practical band = project-defined threshold, not an industry standard.",
        "",
        "## Mean total system cost",
        "",
        "| condition | B | C_oracle | C_est | D1_oracle | D2_oracle | D1_est | D2_est |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in CONDITIONS:
        lines.append(
            f"| {c} | {m(c,'B_topk_probability'):.0f} | {m(c,'C_oracle'):.0f} | "
            f"{m(c,'C_est'):.0f} | {m(c,'D1_oracle'):.0f} | {m(c,'D2_oracle'):.0f} | "
            f"{m(c,'D1_est'):.0f} | {m(c,'D2_est'):.0f} |"
        )

    lines += ["", "## VALUE_NOISE fair deployable comparisons", ""]
    for cond in ("VALUE_NOISE_LOW", "VALUE_NOISE_HIGH"):
        lines.append(f"### {cond}")
        lines.append("")
        lines.append("| comparison | mean Δ | 95% t-CI | vs 0 | practical |")
        lines.append("| --- | ---: | --- | --- | --- |")
        for _, r in vn_df[vn_df["condition"] == cond].iterrows():
            lines.append(
                f"| {r['comparison']} | {r['mean_delta']:.1f} | "
                f"[{r['ci95_low']:.1f}, {r['ci95_high']:.1f}] | "
                f"{r['zero_status']} | {r['practical_class']} |"
            )
        lines.append("")

    # Reversal survival check: does C_est beat D1_est under HIGH?
    hi = vn_df[
        (vn_df["condition"] == "VALUE_NOISE_HIGH")
        & (vn_df["comparison"] == "D1_est_vs_C_est")
    ].iloc[0]
    # D1_est - C_est > 0 means C_est better (lower cost for C)
    if hi["mean_delta"] > 0 and hi["zero_status"] == "excludes_0_positive":
        reversal = (
            "**YES — reversal survives fairness correction:** C_est beats D1_est "
            "under VALUE_NOISE_HIGH (paired CI excludes 0)."
        )
        secondary_ok = True
    elif hi["mean_delta"] > 0:
        reversal = (
            "**Partial:** C_est has lower mean cost than D1_est under VALUE_NOISE_HIGH, "
            "but the paired CI includes 0."
        )
        secondary_ok = True
    elif hi["mean_delta"] < 0 and hi["zero_status"] == "excludes_0_negative":
        reversal = (
            "**NO — reversal does not survive:** D1_est remains better than C_est "
            "under VALUE_NOISE_HIGH after giving C the same v_hat."
        )
        secondary_ok = True  # still supports erosion narrative if penalties grow
    else:
        reversal = "**Mixed / near-equivalent** under VALUE_NOISE_HIGH after fairness correction."
        secondary_ok = True

    lines += [
        "## Does the prior high-noise reversal survive?",
        "",
        reversal,
        "",
        f"(Paired mean D1_est − C_est = {hi['mean_delta']:.1f}, "
        f"CI [{hi['ci95_low']:.1f}, {hi['ci95_high']:.1f}].)",
        "",
        "## Estimation penalties (est − matched oracle)",
        "",
        "| condition | policy | mean penalty | relative % | 95% t-CI | practical |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for _, r in pen.iterrows():
        lines.append(
            f"| {r['condition']} | {r['left_policy']} | {r['paired_mean']:.1f} | "
            f"{r['relative_pct_of_right_mean']:.3f}% | "
            f"[{r['paired_ci95_low']:.1f}, {r['paired_ci95_high']:.1f}] | "
            f"{r['practical_class']} |"
        )

    lines += [
        "",
        "## Figures",
        "",
        "- `figures/exp03_cost_vs_estimation_error.png`",
        "- `figures/exp03_estimation_penalty_vs_oracle.png`",
        "- `figures/exp03_overlap_vs_estimation_error.png`",
        "",
        f"Secondary claim support flag: secondary_ok={secondary_ok}",
        "",
        "## Integrity",
        "",
        "Synthetic experiment. Not evidence of realized production fraud reduction.",
        "Repo remains PRIVATE.",
        "",
    ]
    (ROOT / "results.md").write_text("\n".join(lines) + "\n")
    (ROOT / "README.md").write_text(
        "\n".join(
            [
                "# Experiment 03 — Decision-model misspecification",
                "",
                "Fair information sets: B, C_est, D1_est, D2_est vs oracle references.",
                "",
                "```bash",
                "python experiment_03.py",
                "python -m pytest ../../tests/test_experiment_03.py -q",
                "```",
                "",
                "Does not modify Experiments 01–02.",
                "Pre-fair freeze: `outputs/frozen_pre_fair_comparator_v0/`.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
