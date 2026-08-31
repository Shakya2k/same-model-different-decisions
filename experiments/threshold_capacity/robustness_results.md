# Experiment 02 Robustness (Patch 3)

## Framing

Policy sensitivity under fixed predictions — not universal cost-aware dominance.

## Design

- Scenarios: 135
- Seeds/scenario: 10
- Dataset draws: 1350
- Policy evaluations: 6750
- Interval method: normal approx mean ± 1.96·SE (Monte Carlo CLT), n=10 seeds
- Near-equivalence: 1% (pre-specified); sensitivity (0.005, 0.01, 0.02)
- Generative: y from p_base (calibrated by construction); p(T)=sigmoid(logit(p_base)/T)

## Primary best counts

best_policy
D1_cost_aware_topk          73
D2_cost_aware_abstention    42
A_fixed_threshold_fifo      18
C_expected_exposure          2

## Status counts

status
near_equivalent             68
clear_winner                54
near_equivalent_unstable    13

Integrity notices apply (synthetic; stylized objective).
