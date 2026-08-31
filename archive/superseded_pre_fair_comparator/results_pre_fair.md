# Experiment 03 — Decision-model misspecification

**Status:** PRIVATE. Additive to methodology baseline v0.1 (Exp01–02 frozen).

## Research question

How robust are decision-aware policies when action-value quantities are estimated imperfectly?

## Design

- Seeds: **20** (paired within seed across policies/conditions)
- Capacity K = **2000**
- e_true = **0.8** for evaluation in all conditions
- r_true = r_hat = **25.0** (review cost not misspecified here)
- Evaluation always uses TRUE v, f, e_true, r_true
- D1/D2-estimated rank using v_hat, f_hat, e_hat, r_hat
- B and C use true probability / exposure quantities as stable comparators
- D1/D2-oracle use true cost parameters (reference benchmark)

Noise parameters (mean-one lognormal): see `outputs/noise_parameters.json`.

Paired intervals: Student-t, df = 19, t_crit ≈ 2.093024 (not 1.96).

## Conditions

- `ORACLE`
- `VALUE_NOISE_LOW`
- `VALUE_NOISE_HIGH`
- `FP_COST_NOISE_LOW`
- `FP_COST_NOISE_HIGH`
- `REVIEW_EFFECTIVENESS_OPTIMISM`
- `REVIEW_EFFECTIVENESS_PESSIMISM`
- `JOINT_MODERATE_MISSPECIFICATION`

## Mean total system cost (by condition × policy)

| condition | B | C | D1-oracle | D2-oracle | D1-est | D2-est |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORACLE | 655775 | 583271 | 577821 | 575894 | 577821 | 575894 |
| VALUE_NOISE_LOW | 655775 | 583271 | 577821 | 575894 | 579677 | 577403 |
| VALUE_NOISE_HIGH | 655775 | 583271 | 577821 | 575894 | 589112 | 585977 |
| FP_COST_NOISE_LOW | 655775 | 583271 | 577821 | 575894 | 579535 | 576568 |
| FP_COST_NOISE_HIGH | 655775 | 583271 | 577821 | 575894 | 583423 | 579163 |
| REVIEW_EFFECTIVENESS_OPTIMISM | 655775 | 583271 | 577821 | 575894 | 578187 | 578140 |
| REVIEW_EFFECTIVENESS_PESSIMISM | 655775 | 583271 | 577821 | 575894 | 578269 | 579322 |
| JOINT_MODERATE_MISSPECIFICATION | 655775 | 583271 | 577821 | 575894 | 584008 | 582953 |

## When does a simpler policy beat estimated cost-aware?

Positive `mean_delta_est_minus_simple` ⇒ estimated cost-aware has higher cost
(simpler is better on average).

| condition | estimated | simpler | mean Δ | 95% t-CI | clearer? |
| --- | --- | --- | ---: | --- | --- |
| VALUE_NOISE_HIGH | D1_estimated | C_expected_exposure | 5841.5 | [3806.3, 7876.7] | True |
| VALUE_NOISE_HIGH | D2_estimated | C_expected_exposure | 2706.0 | [339.7, 5072.4] | True |
| FP_COST_NOISE_HIGH | D1_estimated | C_expected_exposure | 152.5 | [-1344.9, 1649.9] | False |
| JOINT_MODERATE_MISSPECIFICATION | D1_estimated | C_expected_exposure | 736.8 | [-735.9, 2209.6] | False |

Rows where simpler mean-beats estimated: **4** / 32
Rows where CI excludes 0 in favor of simpler: **2** / 32

## Publication question (preliminary)

At what level/type of decision-model error does cost-aware advantage erode?
See paired deltas CSV and figures; do not tune to preserve superiority.

## Figures

- `figures/exp03_cost_vs_estimation_error.png`
- `figures/exp03_regret_vs_oracle.png`
- `figures/exp03_overlap_vs_estimation_error.png`

## Integrity

Synthetic experiment. Not evidence of realized production fraud reduction.
Repo remains PRIVATE.

