# Boundary-regime results

## Research question

When should probability ranking, expected-exposure ranking, and cost-aware ranking converge?

## Theory

If transaction value v, false-positive cost f, and review cost r are constant, then

expected benefit = p·v − r − (1−p)·f = p·(v+f) − (r+f)

is strictly increasing in p. Therefore B, C, and D induce the same ordering (and the same top-K set when D does not drop negative-benefit items).

## Homogeneous constant v & f (seed 42)

- B/C/D mask identical: **False**
- Top-K all positive benefit: **False**
- Rank corr p vs exposure: 1.0
- Rank corr p vs benefit: 1.0
- Costs B/C/D: 516520.0 / 516520.0 / 505495.0

If masks are not identical while top-K benefits are all positive, investigate implementation.

## Full table

See `outputs/boundary_regimes.csv`.

## Note on mask inequality under homogeneity

B/C/D **rankings** coincide (overlap 1.0, rank correlation 1.0). Masks can still differ because Policy D does **not** force capacity utilization when expected benefit is non-positive. In the homogeneous seed-42 draw, some of the top-K probability mass had non-positive expected benefit at (v=100,f=40,r=25), so D reviewed fewer cases than B/C. This is consistent with the theory for *ordering*, and intentional for *selection*.
