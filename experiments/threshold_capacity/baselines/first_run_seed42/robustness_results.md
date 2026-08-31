# Experiment 02 Robustness Results

## Integrity notice

This is a synthetic experiment, not evidence of realized fraud reduction in a production financial-services environment. Results depend on the simulated fraud prevalence, probability calibration, transaction-value distribution, intervention costs and capacity assumptions.

## Grid

- Capacity: [500, 1000, 2000, 5000, 10000]
- Prevalence: ['low', 'baseline', 'high']
- Calibration: ['calibrated', 'moderate_distortion', 'severe_distortion']
- FP cost: ['low', 'baseline', 'high']
- Cells: 135
- Seed base: 42

## Win frequency (best total system cost)

| Policy | Wins | Win rate |
| --- | ---: | ---: |
| D Cost-aware | 104 | 0.770 |
| C Expected exposure | 31 | 0.230 |

## Where rankings change

- Across this pre-specified grid, only C or D achieved lowest total system cost.
- Probability-only policies (A/B) did **not** win any cell under these assumptions — a boundary result for H1's contrast case.
- 82 cells flagged for near-ties and related notes: `outputs/experiment_02_adverse_null.csv`.
- Rankings between C and D shift with FP-cost regime and capacity (see heatmap).

## Required figure

`figures/best_policy_heatmap.png` — Best Decision Policy by Review Capacity × False-Positive Cost (baseline prevalence, calibrated).

## Optional

`outputs/experiment_02_prob_rank_vs_best.csv` — relative cost of probability ranking vs best policy.

## Limitations

Same as Experiment 01, plus: calibration distortion is a controlled approximation; prevalence intercepts are design choices, not empirical estimates.
