# Methodology (v0.1)

## Question

Holding predictive scores fixed, how do review-selection policies change simulated system cost under capacity, cost, calibration, abstention, and imperfect economic estimates?

## Generative model (Exp 01–02 baseline)

1. Form `p_base = sigmoid(α + βz + ε)`.
2. Draw `y ~ Bernoulli(p_base)` (calibrated by construction).
3. Policies may see `p(T) = sigmoid(logit(p_base)/T)`. Same `y` across temperatures.
4. Stylized draws stand in for economic loss exposure `v` and false-positive friction `f`.

Frozen as tag `research-baseline-v0.1` for Experiments 01–02. Do not retune after seeing outcomes.

## Policies (Exp 01–02)

| Code | Rule |
| --- | --- |
| A | Fixed threshold + FIFO, capacity K |
| B | Top-K by probability |
| C | Top-K by `p·e·v` |
| D1 | Top-K by `p·e·v − r − (1−p)·f` |
| D2 | D1 order; keep only cases with benefit > 0 |

## Review effectiveness

`e` = fraction of fraud loss prevented conditional on review (deterministic).  
Reviewed residual loss for fraud = `(1−e)·v`.

## Experiment 03 — fair misspecification

- Evaluation: true `v`, `f`, `e_true`, `r_true` only.
- Deployable: `B`, `C_est` (`p·v_hat`), `D1_est`, `D2_est`.
- Oracle references: `C_oracle`, `D1_oracle`, `D2_oracle`.
- Under value noise, `C_est` / `D1_est` / `D2_est` share the same `v_hat`.
- Mean-one lognormal multiplicative noise; LOW ≈ 20% CV, HIGH ≈ 50% CV. Parameters in `experiments/decision_misspecification/outputs/noise_parameters.json`.

## Uncertainty

- Exp02 paired comparisons: Student-t, n=10, df=9.
- Exp03 paired comparisons: Student-t, n=20, df=19.
- 1% practical / near-equivalence band is project-defined, not an industry standard.

## Corrections retained for provenance

- Threshold baseline changed to FIFO so it does not collapse into top-k under a hard capacity.
- Labels drawn from `p_base`, not from the temperature-distorted score.
- Exp03 unequal-information comparator superseded; see `archive/superseded_pre_fair_comparator/`.
