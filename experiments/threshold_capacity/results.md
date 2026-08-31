# Experiment 01 Results (Patch 3)

## Framing

Holding model predictions fixed does not hold operational outcomes fixed. Downstream
policy choices — ranking, action costs, capacity, calibration, and abstention —
can materially change system performance.

## Integrity

This is a synthetic experiment, not evidence of realized fraud reduction in a production financial-services environment. Results depend on the simulated fraud prevalence, probability calibration, transaction-value distribution, intervention costs and capacity assumptions.

The cost-aware policies are evaluated against the same stylized cost structure used to define expected action value. Advantages reflect this decision problem and are not universal dominance.

## Generative model (Patch 3)

y ~ Bernoulli(p_base) with p_base = sigmoid(α + βz + ε) — **calibrated by construction**.
Policies use p(T)=sigmoid(logit(p_base)/T). Same y across T.

## Policies

A FIFO threshold · B top-K p · C top-K p·e·v · D1 cost-aware top-K · D2 cost-aware + abstention

## Canonical table (seed 42, T=1, e=1, K=2000)

| Policy | Volume | Precision | Total cost |
| --- | ---: | ---: | ---: |
| A Fixed threshold + FIFO | 896 | 0.6138 | 610,480.05 |
| B Top-k probability | 2000 | 0.5195 | 606,927.53 |
| C Expected exposure | 2000 | 0.3105 | 521,769.61 |
| D1 Cost-aware top-K | 2000 | 0.3370 | 516,336.73 |
| D2 Cost-aware + abstention | 2000 | 0.3370 | 516,336.73 |

## Attribution

                  metric    value
    ranking_gain_B_to_D1 90590.80
    ranking_gain_C_to_D1  5432.88
abstention_gain_D1_to_D2     0.00

## Predictive metrics by T

 roc_auc    brier  log_loss   T                      label
0.858893 0.042480  0.157690 1.0 calibrated_by_construction
0.858893 0.045480  0.179217 1.5           temperature_T1.5
0.858893 0.063706  0.259644 2.5           temperature_T2.5

Patch-2 baselines preserved under `baselines/patch2_seed42_design/` (superseded for calibration).
