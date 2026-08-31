#!/usr/bin/env python3
"""Boundary sanity + positive-EV homogeneous mask identity for B/C/D1/D2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from decision_model import (
    LABELS,
    GenerativeParams,
    apply_temperature,
    evaluate_policy,
    expected_benefit,
    generate_base_dataset,
    select_for_review,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEED = 42
CAPACITY = 500  # smaller K increases chance all top-K benefits are positive


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # Homogeneous with high value so top-K benefits are positive
    params = GenerativeParams(constant_value=500.0, constant_fp=20.0, review_cost=25.0)
    base = generate_base_dataset(SEED, params)
    df = apply_temperature(base, 1.0)
    p = df["p_hat"].to_numpy()
    benefit = expected_benefit(
        p, df["value"].to_numpy(), df["fp_cost"].to_numpy(), params.review_cost, 1.0
    )
    top = np.argsort(-p, kind="mergesort")[:CAPACITY]
    all_pos = bool((benefit[top] > 0).all())
    masks = {
        pol: select_for_review(df, pol, CAPACITY)  # type: ignore[arg-type]
        for pol in [
            "B_topk_probability",
            "C_expected_exposure",
            "D1_cost_aware_topk",
            "D2_cost_aware_abstention",
        ]
    }
    identical = all(
        np.array_equal(masks["B_topk_probability"], masks[k])
        for k in masks
    )
    rows.append(
        {
            "regime": "homogeneous_high_v_positive_EV_topk",
            "all_topk_positive_benefit": all_pos,
            "bcd1d2_masks_identical": identical,
            "min_topk_benefit": float(benefit[top].min()),
        }
    )

    # Homogeneous modest v where D2 may abstain
    params2 = GenerativeParams(constant_value=100.0, constant_fp=40.0)
    base2 = generate_base_dataset(SEED, params2)
    df2 = apply_temperature(base2, 1.0)
    m_b = select_for_review(df2, "B_topk_probability", 2000)
    m_d1 = select_for_review(df2, "D1_cost_aware_topk", 2000)
    m_d2 = select_for_review(df2, "D2_cost_aware_abstention", 2000)
    benefit2 = expected_benefit(
        df2["p_hat"].to_numpy(),
        df2["value"].to_numpy(),
        df2["fp_cost"].to_numpy(),
        25.0,
        1.0,
    )
    rows.append(
        {
            "regime": "homogeneous_modest_v_abstention_possible",
            "B_volume": int(m_b.sum()),
            "D1_volume": int(m_d1.sum()),
            "D2_volume": int(m_d2.sum()),
            "D1_eq_B": bool(np.array_equal(m_b, m_d1)),
            "D2_subset_positive": bool((~m_d2 | (benefit2 > 0)).all()),
            "rank_corr_p_benefit": float(
                np.corrcoef(
                    np.argsort(np.argsort(-df2["p_hat"].to_numpy())),
                    np.argsort(np.argsort(-benefit2)),
                )[0, 1]
            ),
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "boundary_regimes.csv", index=False)
    md = f"""# Boundary results (Patch 3)

## Ranking convergence (positive-EV top-K)

When v,f,r constant and all top-K cases have positive expected benefit, B/C/D1/D2 masks should match.

Result: all_topk_positive={rows[0]['all_topk_positive_benefit']}, masks_identical={rows[0]['bcd1d2_masks_identical']}, min_topk_benefit={rows[0]['min_topk_benefit']:.4f}

## Abstention divergence

Under modest constant costs, D2 may select fewer than K when some ranked benefits are ≤0, while D1 forces capacity. Ranking of benefit still aligns with p when v,f constant (rank_corr≈1).

See `outputs/boundary_regimes.csv`.
"""
    (ROOT / "boundary_results.md").write_text(md)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
