# Experiment 02 — paired cost differences (additive reporting)

**Baseline:** internal methodology v0.1 (Experiments 01–02 frozen).

Policies are evaluated on the **same generated dataset** within each seed/scenario.
Deltas are per-seed paired differences; uncertainty uses Student-t intervals
(n = 10 seeds, df = 9, t_crit ≈ 2.262157).

Historical Patch-3 normal-approx (±1.96 SE) summaries remain unchanged.
Publication-facing reporting should prefer this paired analysis.

## Comparisons

- `delta_B_D1 = Cost_B − Cost_D1`
- `delta_C_D1 = Cost_C − Cost_D1`
- `delta_D1_D2 = Cost_D1 − Cost_D2`
- `delta_A_D2 = Cost_A − Cost_D2`

Positive delta ⇒ left policy has higher (worse) cost than right.

## Practical-equivalence band

The project's **pre-specified** 1% band is applied to the paired mean/CI
relative to the mean cost of the right-hand (reference) policy.
This is not framed as null-hypothesis significance testing.

Scenarios summarized: **135**. Full table: `outputs/experiment_02_paired_deltas.csv`.

## Headline counts (across scenarios × comparisons)

### delta_B_D1

- CI excludes 0 (left worse): **135** / 135
- CI excludes 0 (left better): **0** / 135
- CI includes 0: **0** / 135
- Entire CI within 1% practical band: **0** / 135

### delta_C_D1

- CI excludes 0 (left worse): **118** / 135
- CI excludes 0 (left better): **0** / 135
- CI includes 0: **17** / 135
- Entire CI within 1% practical band: **45** / 135

### delta_D1_D2

- CI excludes 0 (left worse): **47** / 135
- CI excludes 0 (left better): **0** / 135
- CI includes 0: **88** / 135
- Entire CI within 1% practical band: **90** / 135

### delta_A_D2

- CI excludes 0 (left worse): **115** / 135
- CI excludes 0 (left better): **16** / 135
- CI includes 0: **4** / 135
- Entire CI within 1% practical band: **0** / 135

## Example scenario (baseline prevalence, T=1, FP baseline, K=2000)

| comparison | mean Δ | SE | 95% t-interval | vs 0 | CI in 1% band |
| --- | ---: | ---: | --- | --- | --- |
| delta_B_D1 | 93666.04 | 3593.22 | [85537.61, 101794.47] | excludes_0_positive | False |
| delta_C_D1 | 5617.54 | 543.03 | [4389.11, 6845.96] | excludes_0_positive | False |
| delta_D1_D2 | -29.73 | 29.73 | [-96.98, 37.52] | includes_0 | True |
| delta_A_D2 | 95379.31 | 2889.34 | [88843.16, 101915.46] | excludes_0_positive | False |

Source CSV: `outputs/experiment_02_full_multiseed.csv` (unchanged).

