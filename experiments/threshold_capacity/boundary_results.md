# Boundary results (Patch 3)

## Ranking convergence (positive-EV top-K)

When v,f,r constant and all top-K cases have positive expected benefit, B/C/D1/D2 masks should match.

Result: all_topk_positive=True, masks_identical=True, min_topk_benefit=260.7807

## Abstention divergence

Under modest constant costs, D2 may select fewer than K when some ranked benefits are ≤0, while D1 forces capacity. Ranking of benefit still aligns with p when v,f constant (rank_corr≈1).

See `outputs/boundary_regimes.csv`.
