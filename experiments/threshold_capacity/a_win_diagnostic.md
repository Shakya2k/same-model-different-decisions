# A-win diagnostic (Patch 3)

A (fixed threshold + FIFO) was primary best-by-mean in **18** / 135 scenarios.

## Mechanism summary

Compare mean review volumes: A typically reviews **fewer** cases than forced top-K policies when the mass above p>0.5 is limited, lowering review cost and FP friction while accepting higher residual fraud loss. Wins are concentrated where friction/review costs outweigh residual loss under the stylized objective — not treated as implementation errors.

See `outputs/experiment_02_a_win_diagnostic.csv`.

Mean A volume (A-win cells): 707.8
Mean B volume (same cells): 5972.2
