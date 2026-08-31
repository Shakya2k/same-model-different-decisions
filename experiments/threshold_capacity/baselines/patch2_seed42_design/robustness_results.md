# Experiment 02 Robustness Results (Patch 2)

## Integrity notice

This is a synthetic experiment, not evidence of realized fraud reduction in a production financial-services environment. Results depend on the simulated fraud prevalence, probability calibration, transaction-value distribution, intervention costs and capacity assumptions.

The cost-aware policy is evaluated against the same stylized cost structure used to define its expected action value. Its advantage therefore reflects the assumptions of this decision problem and should not be interpreted as universal dominance.

## Pre-specified near-equivalence

Policies within **1%** of best mean total cost are practically near-equivalent for headline comparison.
Sensitivity bands also reported at 0.5%, 1.0%, 2.0%.

## Design

- Scenarios (operating cells): 135
- Seeds per scenario: 10
- Total simulation runs (scenario × seed × recorded as policy rows / 4): 1,350 dataset draws; 5,400 policy evaluations
- Calibration: temperature scaling on score log-odds — {'calibrated_T1.0': 1.0, 'moderate_T1.5': 1.5, 'severe_T2.5': 2.5}
- Capacity: [500, 1000, 2000, 5000, 10000]
- Prevalence intercepts: {'low': -4.6, 'baseline': -3.9, 'high': -3.2}
- FP mean regimes: {'low': 15.0, 'baseline': 40.0, 'high': 120.0}

## Win / status summary (by mean cost across seeds)

| Policy | Primary best-count |
| --- | ---: |
| D Cost-aware | 112 |
| A Fixed threshold + FIFO | 19 |
| C Expected exposure | 4 |

### Scenario status counts

- clear_winner: 79
- near_equivalent: 42
- near_equivalent_unstable: 11
- unstable_across_seeds: 3

## Effect sizes

See `outputs/experiment_02_effect_sizes.csv` for absolute and % gaps vs best, and `experiment_02_scenario_headlines.csv` for clear_winner / near_equivalent / unstable labels.

## Figures

- `figures/best_policy_heatmap.png`
- `figures/mean_cost_multiseed_canonical_cell.png`

## First-run baseline

Patch-1 single-seed win counts preserved under `baselines/first_run_seed42/` and are not directly comparable to Patch-2 (Policy A redefined; D no longer forces capacity).
