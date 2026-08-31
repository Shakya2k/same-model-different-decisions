# Experiment 01 Results (Patch 2)

## Integrity notice

This is a synthetic experiment, not evidence of realized fraud reduction in a production financial-services environment. Results depend on the simulated fraud prevalence, probability calibration, transaction-value distribution, intervention costs and capacity assumptions.

The cost-aware policy is evaluated against the same stylized cost structure used to define its expected action value. Its advantage therefore reflects the assumptions of this decision problem and should not be interpreted as universal dominance.

## Label

**Illustrative canonical simulation — seed 42.** Generalized conclusions require multi-seed robustness (Experiment 02).

## Cost model

TotalCost = Σ[(1-q_i)·y_i·v_i + q_i·r + q_i·(1-y_i)·f_i]

Simplifying assumption: reviewed fraud is always prevented.

## Policies

- **A**: p_i > 0.50, then FIFO by seeded arrival_index until capacity K
- **B**: top-K by p_i
- **C**: top-K by p_i·v_i
- **D**: rank by expected benefit p·v − r − (1-p)·f; review only positive-benefit items up to K (capacity not forced)

## Design

- N = 100,000
- Seed = 42
- K = 2,000
- r = 25.0
- Empirical prevalence = 0.0517
- Same predictions across policies: Yes
- Policies never read realized y (asserted)

## Table

| Policy | Same predictions? | Review volume | Precision | Fraud loss | FP cost | Review cost | Total cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A Fixed threshold + FIFO | Yes | 867 | 0.5813 | 549,613.13 | 15,876.42 | 21,675.00 | 587,164.55 |
| B Top-k probability | Yes | 2,000 | 0.4660 | 497,147.11 | 46,579.13 | 50,000.00 | 593,726.23 |
| C Expected exposure | Yes | 2,000 | 0.2765 | 416,993.74 | 62,731.51 | 50,000.00 | 529,725.25 |
| D Cost-aware | Yes | 2,000 | 0.3025 | 419,409.94 | 53,536.32 | 50,000.00 | 522,946.26 |

## Summary

Lowest total system cost under these assumptions: **D Cost-aware** (522,946.26).

First-run Patch-1 baseline preserved under `baselines/first_run_seed42/` (A≈B collapse; superseded design).

## Figure

`figures/total_system_cost_by_policy.png` — labeled illustrative seed-42 only.
