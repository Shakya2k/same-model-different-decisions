# Experiment 01 Results

## Integrity notice

This is a synthetic experiment, not evidence of realized fraud reduction in a production financial-services environment. Results depend on the simulated fraud prevalence, probability calibration, transaction-value distribution, intervention costs and capacity assumptions.

## Design

- N = 100,000
- Seed = 42
- Review capacity = 2,000
- Review cost per case = 25.0
- Empirical fraud prevalence = 0.0517
- **Same predictions across all policies:** Yes

## Assumptions

1. Only reviewed fraud cases have loss captured (avoided); unreviewed fraud remains as loss.
2. False-positive reviews incur case-specific `fp_cost`.
3. Every review incurs fixed operational review cost.
4. No hard-block action beyond review (simplified action set).
5. Scores are held fixed; only ranking / thresholding rules change.

## Table

| Policy | Same predictions? | Review volume | Precision | Fraud loss | FP cost | Review cost | Total cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A Fixed threshold | Yes | 2,000 | 0.4660 | 497,147.11 | 46,579.13 | 50,000.00 | 593,726.23 |
| B Top-k probability | Yes | 2,000 | 0.4660 | 497,147.11 | 46,579.13 | 50,000.00 | 593,726.23 |
| C Expected exposure | Yes | 2,000 | 0.2765 | 416,993.74 | 62,731.51 | 50,000.00 | 529,725.25 |
| D Cost-aware | Yes | 2,000 | 0.3025 | 419,409.94 | 53,536.32 | 50,000.00 | 522,946.26 |

## Summary

Lowest total system cost under these assumptions: **D Cost-aware** (522,946.26).

This ranking is conditional. See Experiment 02 for regimes where probability-only ranking is competitive or preferred.

## Figure

`figures/total_system_cost_by_policy.png`

## Limitations

- Synthetic labels and costs
- No adversary adaptation
- No queue delay dynamics
- Near-calibrated scores by construction in Experiment 01
