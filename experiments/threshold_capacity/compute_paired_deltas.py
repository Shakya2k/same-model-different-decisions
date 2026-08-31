#!/usr/bin/env python3
"""
Additive paired-delta reporting for Experiment 02 (v0.1 baseline).

Does NOT re-run Experiment 02 or alter historical Patch-3 outputs.
Reads outputs/experiment_02_full_multiseed.csv and writes paired summaries.

Uses Student-t 95% intervals with df = n_seeds - 1 (n=10 → df=9).
Does NOT use 1.96 for these paired intervals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
NEAR_EQUIVALENCE_PCT = 0.01  # project's pre-specified practical band
# Exact two-sided 95% critical value for Student-t, df=9
T_CRIT_DF9 = 2.2621571627409915

COMPARISONS = [
    ("delta_B_D1", "B_topk_probability", "D1_cost_aware_topk"),
    ("delta_C_D1", "C_expected_exposure", "D1_cost_aware_topk"),
    ("delta_D1_D2", "D1_cost_aware_topk", "D2_cost_aware_abstention"),
    ("delta_A_D2", "A_fixed_threshold_fifo", "D2_cost_aware_abstention"),
]

SCENARIO_KEYS = ["prevalence", "calibration", "fp_cost_regime", "capacity"]


def paired_stats(deltas: np.ndarray) -> dict:
    n = int(len(deltas))
    mean = float(np.mean(deltas))
    # sample SD (ddof=1); for n=1 guard
    sd = float(np.std(deltas, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    df = n - 1
    t_crit = T_CRIT_DF9 if df == 9 else float("nan")
    if n > 1 and np.isfinite(se):
        lo = mean - t_crit * se
        hi = mean + t_crit * se
    else:
        lo = hi = float("nan")
    # relation to 0
    if lo > 0:
        zero_status = "excludes_0_positive"
    elif hi < 0:
        zero_status = "excludes_0_negative"
    else:
        zero_status = "includes_0"
    # practical-equivalence band around 0 relative to |mean| is not used;
    # band is applied vs reference cost scale: |mean_delta| / mean(|costs|) later.
    return {
        "n_seeds": n,
        "df": df,
        "mean_paired_delta": mean,
        "sd_paired_delta": sd,
        "se_paired_delta": se,
        "t_crit": t_crit,
        "ci95_low": lo,
        "ci95_high": hi,
        "interval_method": "student_t_df_n_minus_1",
        "zero_status": zero_status,
    }


def main() -> None:
    src = OUT / "experiment_02_full_multiseed.csv"
    raw = pd.read_csv(src)

    # Wide costs by policy within seed/scenario
    pivot = raw.pivot_table(
        index=SCENARIO_KEYS + ["seed"],
        columns="policy",
        values="total_system_cost",
        aggfunc="first",
    ).reset_index()

    rows: list[dict] = []
    for delta_name, left, right in COMPARISONS:
        for keys, g in pivot.groupby(SCENARIO_KEYS):
            key_dict = dict(zip(SCENARIO_KEYS, keys if isinstance(keys, tuple) else (keys,)))
            deltas = (g[left] - g[right]).to_numpy(dtype=float)
            stats = paired_stats(deltas)
            # practical band: |CI| and mean relative to mean cost of the right (reference) policy
            ref_mean = float(g[right].mean())
            band = NEAR_EQUIVALENCE_PCT * abs(ref_mean)
            mean_abs = abs(stats["mean_paired_delta"])
            ci_lo, ci_hi = stats["ci95_low"], stats["ci95_high"]
            # "remains within the 1% practical-equivalence band" — entire CI inside [-band, band]
            if np.isfinite(ci_lo) and np.isfinite(ci_hi):
                within_band = (ci_lo >= -band) and (ci_hi <= band)
            else:
                within_band = False
            rows.append(
                {
                    **key_dict,
                    "comparison": delta_name,
                    "left_policy": left,
                    "right_policy": right,
                    **stats,
                    "ref_mean_cost_right": ref_mean,
                    "practical_band_abs": band,
                    "mean_abs_within_1pct_band": mean_abs <= band + 1e-12,
                    "ci_within_1pct_practical_band": within_band,
                    "practical_band_note": (
                        "project pre-specified 1% band of |mean cost of right policy|; "
                        "not an industry standard"
                    ),
                }
            )

    out = pd.DataFrame(rows)
    out_path = OUT / "experiment_02_paired_deltas.csv"
    out.to_csv(out_path, index=False)

    # Markdown summary — publication-facing paired analysis
    n_scen = out.groupby(SCENARIO_KEYS).ngroups
    lines = [
        "# Experiment 02 — paired cost differences (additive reporting)",
        "",
        "**Baseline:** internal methodology v0.1 (Experiments 01–02 frozen).",
        "",
        "Policies are evaluated on the **same generated dataset** within each seed/scenario.",
        "Deltas are per-seed paired differences; uncertainty uses Student-t intervals",
        f"(n = 10 seeds, df = 9, t_crit ≈ {T_CRIT_DF9:.6f}).",
        "",
        "Historical Patch-3 normal-approx (±1.96 SE) summaries remain unchanged.",
        "Publication-facing reporting should prefer this paired analysis.",
        "",
        "## Comparisons",
        "",
        "- `delta_B_D1 = Cost_B − Cost_D1`",
        "- `delta_C_D1 = Cost_C − Cost_D1`",
        "- `delta_D1_D2 = Cost_D1 − Cost_D2`",
        "- `delta_A_D2 = Cost_A − Cost_D2`",
        "",
        "Positive delta ⇒ left policy has higher (worse) cost than right.",
        "",
        "## Practical-equivalence band",
        "",
        "The project's **pre-specified** 1% band is applied to the paired mean/CI",
        "relative to the mean cost of the right-hand (reference) policy.",
        "This is not framed as null-hypothesis significance testing.",
        "",
        f"Scenarios summarized: **{n_scen}**. Full table: `outputs/experiment_02_paired_deltas.csv`.",
        "",
        "## Headline counts (across scenarios × comparisons)",
        "",
    ]

    for delta_name, _, _ in COMPARISONS:
        sub = out[out["comparison"] == delta_name]
        excl_pos = int((sub["zero_status"] == "excludes_0_positive").sum())
        excl_neg = int((sub["zero_status"] == "excludes_0_negative").sum())
        incl = int((sub["zero_status"] == "includes_0").sum())
        within = int(sub["ci_within_1pct_practical_band"].sum())
        lines.append(f"### {delta_name}")
        lines.append("")
        lines.append(f"- CI excludes 0 (left worse): **{excl_pos}** / {len(sub)}")
        lines.append(f"- CI excludes 0 (left better): **{excl_neg}** / {len(sub)}")
        lines.append(f"- CI includes 0: **{incl}** / {len(sub)}")
        lines.append(f"- Entire CI within 1% practical band: **{within}** / {len(sub)}")
        lines.append("")

    # Example: baseline / T1 / baseline FP / K=2000
    eg = out[
        (out["prevalence"] == "baseline")
        & (out["calibration"] == "calibrated_by_construction_T1.0")
        & (out["fp_cost_regime"] == "baseline")
        & (out["capacity"] == 2000)
    ]
    lines.append("## Example scenario (baseline prevalence, T=1, FP baseline, K=2000)")
    lines.append("")
    lines.append("| comparison | mean Δ | SE | 95% t-interval | vs 0 | CI in 1% band |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for _, r in eg.iterrows():
        lines.append(
            f"| {r['comparison']} | {r['mean_paired_delta']:.2f} | {r['se_paired_delta']:.2f} | "
            f"[{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] | {r['zero_status']} | "
            f"{bool(r['ci_within_1pct_practical_band'])} |"
        )
    lines.append("")
    lines.append("Source CSV: `outputs/experiment_02_full_multiseed.csv` (unchanged).")
    lines.append("")

    md_path = ROOT / "paired_comparison_results.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
