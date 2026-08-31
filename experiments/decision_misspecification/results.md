# Experiment 03 — Decision-model misspecification (fair information sets)

**Status:** PRIVATE. Additive to methodology baseline v0.1.
Pre-fair outputs frozen at `outputs/frozen_pre_fair_comparator_v0/`.

## Terminology

- `v` = true **economic loss exposure conditional on fraud** (evaluation oracle)
- `v_hat` = policy estimate of that exposure
- Simulator uses stylized monetary draws; this does **not** reproduce a specific bank's fraud-loss process.

`e` = fraction of fraud loss prevented conditional on review (deterministic fractional model).

## Information hierarchy

| Policy | Information | Deployable? |
| --- | --- | --- |
| B | p only | yes |
| C_est | p + v_hat | yes |
| D1_est | p + v_hat + estimated friction/action economics | yes |
| D2_est | D1_est + abstention | yes |
| C_oracle / D1_oracle / D2_oracle | true quantities | reference only |

Fair value-noise rule: **C_est and D*_est share the same v_hat**.

Seeds=20, K=2000, e_true=0.8, Student-t df=19.
1% practical band = project-defined threshold, not an industry standard.

## Mean total system cost

| condition | B | C_oracle | C_est | D1_oracle | D2_oracle | D1_est | D2_est |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ORACLE | 655775 | 583271 | 583271 | 577821 | 575894 | 577821 | 575894 |
| VALUE_NOISE_LOW | 655775 | 583271 | 585831 | 577821 | 575894 | 579677 | 577403 |
| VALUE_NOISE_HIGH | 655775 | 583271 | 597580 | 577821 | 575894 | 589112 | 585977 |
| FP_COST_NOISE_LOW | 655775 | 583271 | 583271 | 577821 | 575894 | 579535 | 576568 |
| FP_COST_NOISE_HIGH | 655775 | 583271 | 583271 | 577821 | 575894 | 583423 | 579163 |
| REVIEW_EFFECTIVENESS_OPTIMISM | 655775 | 583271 | 583271 | 577821 | 575894 | 578187 | 578140 |
| REVIEW_EFFECTIVENESS_PESSIMISM | 655775 | 583271 | 583271 | 577821 | 575894 | 578269 | 579322 |
| JOINT_MODERATE_MISSPECIFICATION | 655775 | 583271 | 589320 | 577821 | 575894 | 584008 | 582953 |

## VALUE_NOISE fair deployable comparisons

### VALUE_NOISE_LOW

| comparison | mean Δ | 95% t-CI | vs 0 | practical |
| --- | ---: | --- | --- | --- |
| C_est_vs_B | -69944.4 | [-74303.9, -65584.8] | excludes_0_negative | outside_1pct_practical_band |
| D1_est_vs_B | -76098.6 | [-80311.7, -71885.5] | excludes_0_negative | outside_1pct_practical_band |
| D2_est_vs_B | -78371.9 | [-82419.0, -74324.8] | excludes_0_negative | outside_1pct_practical_band |
| D1_est_vs_C_est | -6154.2 | [-7546.0, -4762.5] | excludes_0_negative | outside_1pct_practical_band |
| D2_est_vs_C_est | -8427.6 | [-9796.1, -7059.0] | excludes_0_negative | outside_1pct_practical_band |

### VALUE_NOISE_HIGH

| comparison | mean Δ | 95% t-CI | vs 0 | practical |
| --- | ---: | --- | --- | --- |
| C_est_vs_B | -58194.7 | [-62441.5, -53947.9] | excludes_0_negative | outside_1pct_practical_band |
| D1_est_vs_B | -66663.0 | [-70642.4, -62683.6] | excludes_0_negative | outside_1pct_practical_band |
| D2_est_vs_B | -69798.5 | [-73837.7, -65759.2] | excludes_0_negative | outside_1pct_practical_band |
| D1_est_vs_C_est | -8468.3 | [-9643.0, -7293.6] | excludes_0_negative | outside_1pct_practical_band |
| D2_est_vs_C_est | -11603.7 | [-12866.4, -10341.1] | excludes_0_negative | outside_1pct_practical_band |

## Does the prior high-noise reversal survive?

**NO — reversal does not survive:** D1_est remains better than C_est under VALUE_NOISE_HIGH after giving C the same v_hat.

(Paired mean D1_est − C_est = -8468.3, CI [-9643.0, -7293.6].)

## Estimation penalties (est − matched oracle)

| condition | policy | mean penalty | relative % | 95% t-CI | practical |
| --- | --- | ---: | ---: | --- | --- |
| ORACLE | C_est | 0.0 | 0.000% | [0.0, 0.0] | ci_within_1pct_practical_band |
| ORACLE | D1_est | 0.0 | 0.000% | [0.0, 0.0] | ci_within_1pct_practical_band |
| ORACLE | D2_est | 0.0 | 0.000% | [0.0, 0.0] | ci_within_1pct_practical_band |
| VALUE_NOISE_LOW | C_est | 2560.1 | 0.439% | [1282.1, 3838.1] | ci_within_1pct_practical_band |
| VALUE_NOISE_LOW | D1_est | 1855.7 | 0.321% | [515.6, 3195.9] | ci_within_1pct_practical_band |
| VALUE_NOISE_LOW | D2_est | 1509.2 | 0.262% | [412.4, 2605.9] | ci_within_1pct_practical_band |
| VALUE_NOISE_HIGH | C_est | 14309.8 | 2.453% | [12402.5, 16217.1] | outside_1pct_practical_band |
| VALUE_NOISE_HIGH | D1_est | 11291.3 | 1.954% | [9611.7, 12970.9] | outside_1pct_practical_band |
| VALUE_NOISE_HIGH | D2_est | 10082.7 | 1.751% | [7998.1, 12167.2] | outside_1pct_practical_band |
| FP_COST_NOISE_LOW | C_est | 0.0 | 0.000% | [0.0, 0.0] | ci_within_1pct_practical_band |
| FP_COST_NOISE_LOW | D1_est | 1714.2 | 0.297% | [784.2, 2644.2] | ci_within_1pct_practical_band |
| FP_COST_NOISE_LOW | D2_est | 673.6 | 0.117% | [-129.4, 1476.6] | ci_within_1pct_practical_band |
| FP_COST_NOISE_HIGH | C_est | 0.0 | 0.000% | [0.0, 0.0] | ci_within_1pct_practical_band |
| FP_COST_NOISE_HIGH | D1_est | 5602.3 | 0.970% | [4337.0, 6867.5] | mean_within_1pct_band_ci_not |
| FP_COST_NOISE_HIGH | D2_est | 3268.6 | 0.568% | [1990.6, 4546.6] | ci_within_1pct_practical_band |
| REVIEW_EFFECTIVENESS_OPTIMISM | C_est | 0.0 | 0.000% | [0.0, 0.0] | ci_within_1pct_practical_band |
| REVIEW_EFFECTIVENESS_OPTIMISM | D1_est | 366.1 | 0.063% | [-300.9, 1033.2] | ci_within_1pct_practical_band |
| REVIEW_EFFECTIVENESS_OPTIMISM | D2_est | 2245.9 | 0.390% | [1244.2, 3247.7] | ci_within_1pct_practical_band |
| REVIEW_EFFECTIVENESS_PESSIMISM | C_est | 0.0 | 0.000% | [0.0, 0.0] | ci_within_1pct_practical_band |
| REVIEW_EFFECTIVENESS_PESSIMISM | D1_est | 447.9 | 0.078% | [-301.5, 1197.3] | ci_within_1pct_practical_band |
| REVIEW_EFFECTIVENESS_PESSIMISM | D2_est | 3427.8 | 0.595% | [1828.7, 5026.8] | ci_within_1pct_practical_band |
| JOINT_MODERATE_MISSPECIFICATION | C_est | 6049.2 | 1.037% | [4666.3, 7432.1] | outside_1pct_practical_band |
| JOINT_MODERATE_MISSPECIFICATION | D1_est | 6186.7 | 1.071% | [4991.4, 7381.9] | outside_1pct_practical_band |
| JOINT_MODERATE_MISSPECIFICATION | D2_est | 7059.0 | 1.226% | [5726.8, 8391.3] | outside_1pct_practical_band |

## Figures

- `figures/exp03_cost_vs_estimation_error.png`
- `figures/exp03_estimation_penalty_vs_oracle.png`
- `figures/exp03_overlap_vs_estimation_error.png`

Secondary claim support flag: secondary_ok=True

## Integrity

Synthetic experiment. Not evidence of realized production fraud reduction.
Repo remains PRIVATE.

