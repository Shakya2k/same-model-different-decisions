# Threshold & capacity experiments (01–02)

Synthetic review-selection policies under a hard capacity ceiling. Fraud is the simulation domain only.

## Experiment 01

```bash
python experiment_01.py
```

## Experiment 02

Full grid (heavier: 135 × 10 seeds):

```bash
python experiment_02_robustness.py
```

Paired Student-t summaries from the multiseed CSV:

```bash
python compute_paired_deltas.py
```

Committed publication-facing tables live under `outputs/`. The full multiseed CSV is regenerable and not required to read the article.

Parameters were set before looking at outcomes. Do not retune just to force a preferred winner.
