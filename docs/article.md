# Same Model, Different Decisions

**A Synthetic Study of Policy Sensitivity Under Capacity, Cost, and Misspecification**

*Independent technical project — 2026 · Shakya Bhattacharyya*

---

A model score is not a decision.

Suppose a fraud model assigns the same 100,000 transactions the same risk scores. One team reviews the highest probabilities. Another ranks by expected loss. A third accounts for review cost and customer friction.

The model has not changed. The operational outcome can.

That gap between prediction and action is what I wanted to study. Everything below is synthetic. No production banking data.

## Why the ranking rule matters

I started with a simple question: if the model scores stay fixed, how much can the downstream decision rule change the result?

The simulation draws labels from a calibrated base probability, optionally distorts the scores shown to policies with a temperature transform, and compares review-selection policies under a hard capacity ceiling. Economic loss exposure `v` is a stylized severity conditional on fraud — not a claim about any bank's true loss process.

Policies:

- A — fixed threshold + FIFO (capacity-capped)
- B — top-K by probability
- C — top-K by expected exposure
- D1 — top-K by expected benefit accounting for review cost and false-positive friction
- D2 — D1 ranking, but abstain when estimated benefit ≤ 0

Cost-aware policies are not treated as universally better. They are optimized against the same stylized economics used in evaluation, which gives them a structural advantage in the oracle setting.

## Experiment 01 — same predictions, different policies

At the canonical seed (T=1, e=1, K=2000), top-k probability produced a simulated system cost of about $606.9k, compared with about $516.3k for cost-aware top-k — about $90.6k lower on this seed. Expected-exposure ranking landed in between.

On that draw, D1 and D2 matched: every forced top-K case still had positive benefit, so abstention did nothing.

## When capacity and costs change the answer

Experiment 02 sweeps capacity, prevalence, calibration temperature, and false-positive cost (135 cells × 10 seeds).

Cost-aware ranking was expected to beat probability ranking when values and friction costs vary. That mostly happened under the oracle setup, but the difference was often small. Only 54 of 135 cells had a clear winner under the project's 1% near-equivalence rule. In 18 cells, the fixed-threshold policy was cheaper because it reviewed far fewer cases.

Across that grid, under correctly specified economics, the paired mean-cost gap between probability-only ranking and oracle cost-aware ranking favored the cost-aware policy in every cell. That comparison assumes the quantities used to value interventions are correct. Experiment 03 relaxes that.

## Calibration changes probabilities, not ranking

Temperature scaling left AUC unchanged but altered probability magnitudes. Policies that used those magnitudes therefore changed even though the ranking did not. Brier score and log loss got worse as temperature rose.

## Ranking versus abstention

D2 can refuse negative-EV reviews. On the high-effectiveness canonical cell it often ties D1. Separation shows up when effectiveness or value regimes make marginal reviews unattractive.

## What the oracle policies assume

Oracle C/D policies use true `v`, `f`, `e`, and `r`. In a deployed system those quantities are estimated. Oracle results are reference bounds.

## Experiment 03 — imperfect economic estimates

Evaluation always uses true quantities. Deployable policies use estimates:

- B — probability only
- C_est — probability + estimated exposure
- D1_est — + estimated friction/action economics
- D2_est — + abstention

Under the pre-specified value-noise conditions, estimated cost-aware policies retained lower mean simulated cost than both `C_est` and `B`, while estimation penalties versus matched oracles increased as noise grew (about 1.8–2.5% under high value noise for the matched pairs). Within those tested noise levels, the fair comparison did not reverse in favor of estimated-exposure ranking.

### The unfair comparison

An earlier version of Experiment 03 gave expected-exposure ranking access to true economic exposure while the estimated cost-aware policies used noisy estimates. That was not a fair comparison for the question I wanted to answer. After equalizing the information set, the apparent high-noise reversal disappeared. The original run is kept in `archive/superseded_pre_fair_comparator/` rather than deleted.

I initially implemented the threshold baseline in a way that effectively collapsed it into top-k ranking under the capacity constraint, so I changed it to FIFO. Calibration labeling also needed a correction earlier: drawing labels from a noisy score is not the same as drawing them from the probability the policy sees.

## Where this simulation stops being realistic

Review effectiveness is modeled as a deterministic fraction of prevented loss. Real review systems are stochastic and operationally more complicated. The simplified form is useful here because it isolates the policy mechanics, not because it is intended as a faithful model of investigation outcomes.

There is also no adversarial adaptation, no delayed labels, no queueing, no fairness analysis, and no causal claim. Simulated dollars are not realized dollars. Full list: `docs/limitations.md`.

## Reproduction

See the root README. Seeds are fixed. No private data.

## Citation

Bhattacharyya, Shakya. "Same Model, Different Decisions: A Synthetic Study of Policy Sensitivity Under Capacity, Cost, and Misspecification." Independent technical project, 2026.
