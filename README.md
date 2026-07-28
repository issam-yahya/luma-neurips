# LUMA — standalone

Self-contained package to train and reproduce **LUMA** (the `LUMA` model)
for long-term time-series forecasting. It follows the
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) (thuml)
directory layout, so any TSLib workflow works here unchanged.

```
LUMA/
├── run.py                # entry point (unchanged TSLib CLI)
├── requirements.txt
├── data_provider/        # dataset loaders / factory
├── exp/                  # experiment loops (long/short-term, imputation, ...)
├── layers/               # layer building blocks used by the models
├── models/               # model zoo — LUMA is models/LUMA.py
├── utils/                # metrics, tools, time features
├── scripts/              # TSLib-style run scripts
├── dataset/              # the 6 datasets LUMA is evaluated on (bundled)
└── checkpoints/          # (created at run time)
```

## Setup

```bash
conda create -n luma python=3.10 -y
conda activate luma
pip install -r requirements.txt
```

## Quick start — train LUMA on one setting

```bash
python run.py \
  --task_name long_term_forecast --is_training 1 \
  --model LUMA --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 720 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 --d_model 512 --dropout 0.001 \
  --period_len 24 --rank 3 \
  --learning_rate 0.02 --train_epochs 30 --batch_size 256 --patience 5 \
  --lradj type1 --loss MSE --seed 2021
```

`--period_len` (P) and `--rank` (R) are LUMA's two structural knobs.

## Datasets

`dataset/` ships empty. Populate it with the six datasets LUMA is evaluated
on: `ETT-small` (ETTh1/h2, ETTm1/m2), `electricity`, `traffic`,
`exchange_rate`, `weather`, `illness`. Get them from
[Time-Series-Library](https://github.com/thuml/Time-Series-Library). Drop any
additional TSLib-format dataset into `dataset/` to train on it.

## The model

`models/LUMA.py` — a minimal low-rank forecaster. The core is a per-period
low-rank linear map `W = U · diag(s) · Vᵀ` applied over length-`period_len`
segments of the input, with rank `R` controlling capacity. It depends only on
`torch`/`numpy` (no `layers/` imports), so it is easy to lift out on its own.
