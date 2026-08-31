# Internal methodology baseline v0.1

**Tag name:** `research-baseline-v0.1`  
**Scope:** Experiments 01–02 (Patch 3), private repository only.  
**Not a public GitHub Release.**

## Freeze rules

Do not alter:

- canonical seed / seed construction for Exp01–02
- underlying distributions
- policy definitions A / B / C / D1 / D2
- temperature grid, capacity grid, prevalence grid, FP-cost grid
- near-equivalence threshold (1% pre-specified project band)
- historical Exp01–02 result CSVs and figures under `experiments/threshold_capacity/`

Additive reporting (e.g. paired Student-t deltas on the existing multiseed CSV) is allowed without rewriting historical Patch-3 summaries.

## Review-effectiveness semantics (wording only)

`e` = fraction of fraud loss prevented conditional on review (deterministic).  
Residual reviewed fraud loss = `(1-e)*v` for fraud cases that are reviewed.

## Next work

Experiment 03 — decision-model misspecification — lives under `experiments/decision_misspecification/` and must not mutate Exp01–02 artifacts.
