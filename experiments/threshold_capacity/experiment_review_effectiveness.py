#!/usr/bin/env python3
"""Targeted review-effectiveness sensitivity (e in {0.5, 0.8, 1.0})."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from decision_model import (
    LABELS,
    GenerativeParams,
    apply_temperature,
    evaluate_policy,
    generate_base_dataset,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEED = 42
CAPACITY = 2_000
EFF = [0.5, 0.8, 1.0]
POLICIES = [
    "B_topk_probability",
    "C_expected_exposure",
    "D1_cost_aware_topk",
    "D2_cost_aware_abstention",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = generate_base_dataset(SEED, GenerativeParams())
    df = apply_temperature(base, 1.0)
    rows = []
    for e in EFF:
        for p in POLICIES:
            r = evaluate_policy(
                df, p, CAPACITY, review_effectiveness=e  # type: ignore[arg-type]
            )
            r["e"] = e
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "review_effectiveness_sensitivity.csv", index=False)
    md = """# Review-effectiveness sensitivity

Canonical seed-42, T=1, K=2000, baseline generative params.

e = fraction of fraud loss prevented conditional on review.

benefit = p·e·v − r − (1−p)·f

Realized residual reviewed-fraud loss = q·y·(1−e)·v

See `outputs/review_effectiveness_sensitivity.csv`.
"""
    (ROOT / "review_effectiveness_results.md").write_text(md)
    print(
        out.pivot_table(
            index="e", columns="policy", values="total_system_cost"
        ).to_string()
    )


if __name__ == "__main__":
    main()
