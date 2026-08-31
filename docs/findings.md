# Findings (v0.1)

## Central claim

Prediction quality alone does not determine system performance. In this synthetic study, downstream ranking, capacity, intervention economics, abstention, calibration, and uncertainty in economic estimates materially affected the outcomes produced from the same predictive signals.

## Secondary finding (fair Experiment 03)

Richer decision policies benefited from more complete economic information in this synthetic setting, but misspecification imposed measurable costs relative to their oracle counterparts. Under the pre-specified value-noise conditions tested here, the estimated cost-aware policies retained lower mean system cost than both estimated-exposure and probability-only ranking, while their distance from oracle performance increased as estimation error grew.

Do not claim that richer policies became worse than simpler ones under fair VALUE_NOISE_HIGH, that severe misspecification reversed that comparison, or that cost-aware decisioning is universally superior.

## What held

- Same predictions produced different costs across policies (Exp01).
- Under correctly specified economics, oracle cost-aware ranking beat probability-only ranking across the Exp02 grid (paired analysis).
- Fair value noise: `D1_est` / `D2_est` remained cheaper than `C_est` and `B` in the tested conditions.
- Oracle Exp03 condition reproduces equality of matched oracle and estimated policies.

## What did not go as expected

- Threshold+FIFO won 18/135 Exp02 cells by reviewing fewer cases.
- High near-equivalence rates — winner counts alone overstate separation.
- D1 vs D2 often tied when top-K benefits were all positive.
- Temperature worsened probability metrics without changing ranks/AUC.
- An unfair Exp03 information set created an apparent high-noise reversal; after equalizing `v_hat`, it disappeared (archived).
- Estimation penalties grew with value-noise severity.
