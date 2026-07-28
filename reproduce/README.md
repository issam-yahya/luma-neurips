# Reproducing LUMA / LUMA results

This directory reproduces every reported best-configuration result by re-running
`run.py` with the **exact** hyperparameters recorded in the original training
logs. It is fully self-contained: no manual editing, no hidden state.

## Contents

| file | purpose |
|---|---|
| `configs_luma_best.csv` | exact hyperparameters + reference MSE/MAE for all 36 configs |
| `reproduce_luma.sh` | driver — re-runs `run.py` for each config |
| `check_reproduction.py` | verifier — compares fresh results to the reference |

## Requirements

- The `luma` conda environment (PyTorch, numpy, pandas).
- Datasets present under `./dataset/` (ETT-small, electricity, traffic,
  exchange_rate, weather, illness) — the standard Time-Series-Library layout.
- One CUDA GPU (CPU works but is slow).

## Run it

```bash
conda activate luma

# everything (36 configs, ~1–2 h on one GPU)
bash reproduce/reproduce_luma.sh

# or a subset
bash reproduce/reproduce_luma.sh ETTh1 ETTh2

# pick a GPU
GPU=1 bash reproduce/reproduce_luma.sh
```

Per-config logs land in `logs_reproduce/`, and a summary table in
`logs_reproduce/reproduced_results.csv`.

## Verify

```bash
python reproduce/check_reproduction.py            # 2% tolerance (default)
python reproduce/check_reproduction.py --tol 0.01 # stricter
```

It prints a per-config `ref_mse` vs `repro_mse` comparison and an overall
**PASS/FAIL** (exit code 0 on full pass, 1 otherwise).

## The seed matters — read this

The reported numbers were produced with **seed 2021** (`reproduce_luma.sh`
defaults to it). This is not run.py's own default (2); using the wrong seed
gives visibly different numbers, because at low rank the orthogonal
initialization of the low-rank map `W = U diag(s) Vᵀ` is genuinely
seed-sensitive.

- With **seed 2021** reproduction is essentially bit-exact (verified: ETTh1
  matches the reference to 6 decimal places).
- Across **seeds 2021–2025** MSE varies only ~0.3%, so the results are stable —
  the headline number is the seed-2021 draw, not a cherry-picked outlier.

To reproduce the reported table exactly, keep `SEED=2021`. To check stability,
sweep it:

```bash
for s in 2021 2022 2023 2024 2025; do
  SEED=$s bash reproduce/reproduce_luma.sh ETTh2
done
```

## What "reproduce" covers

`configs_luma_best.csv` is generated directly from the original logs, so the
period (`P`), rank (`R`), sequence length, learning rate, batch size, dropout,
and epoch budget for each `(dataset, pred_len)` are exactly what produced the
reported numbers. Fixed across all runs: `--lradj type1`, `--loss MSE`,
`--label_len 48`, `--d_model 512`, `--features M`, 5-epoch early-stopping
patience.
