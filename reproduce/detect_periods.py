"""
FFT-based period detection, restricted to the datasets where it actually
matches the reported best period_len (P): ETTh1 and traffic.

This is the same _fft_period_length algorithm that lives commented-out in
models/MinimalTS.py, applied to non-overlapping seq_len windows of each
dataset's train split (mirroring what the model would see at inference).

Across all 9 datasets in reproduce/configs_luma_best.csv, FFT-detected period
matches the reported best P in only 11/36 configs -- reliably so only for
ETTh1 (4/4) and traffic (4/4); electricity partially matches (3/4); ETTh2,
ETTm1, ETTm2, exchange, weather, ili do not match. See fft_vs_bestP.py at the
repo root for the full 36-row comparison.

Output: scripts/reproduce/detected_periods.json, containing only the
datasets/pred_lens where fft_P == best_P.
"""
import os
import json
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'dataset')
CONFIGS_CSV = os.path.join(ROOT, 'reproduce', 'configs_luma_best.csv')
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detected_periods.json')

# Only datasets where FFT period detection matches the reported best P.
KEEP_DATASETS = {'ETTh1', 'traffic'}

DATASET_PATHS = {
    'ETTh1':   ('ETT-small/ETTh1.csv', 'ETTh'),
    'traffic': ('traffic/traffic.csv', 'ratio'),
}


def load_train_split(path, convention):
    df = pd.read_csv(path)
    cols = list(df.columns)
    if 'date' in cols:
        cols.remove('date')
    data = df[cols].values.astype(np.float64)
    if convention == 'ETTh':
        border2 = 12 * 30 * 24
    else:
        border2 = int(len(df) * 0.7)
    return data[0:border2]


def fft_period_length(x_mean_t: torch.Tensor, L: int, min_w: int, max_w: int) -> int:
    """Verbatim port of the commented-out _fft_period_length in models/MinimalTS.py."""
    spec = torch.fft.rfft(x_mean_t, dim=1)
    power = (spec.real ** 2 + spec.imag ** 2)
    power[:, 0] = 0

    freqs = torch.arange(power.size(1), device=x_mean_t.device)

    with torch.no_grad():
        period_est = torch.where(
            freqs > 0,
            (L / torch.clamp(freqs, min=1)).round().long(),
            torch.zeros_like(freqs)
        )
        valid = (period_est >= min_w) & (period_est <= max_w)
        masked_power = torch.where(valid, power, torch.zeros_like(power))

    k_hat = masked_power.argmax(dim=1)
    w_hat_b = (L / torch.clamp(k_hat, min=1)).round().long()
    w_hat_b = torch.clamp(w_hat_b, min=min_w, max=max_w)

    w_hat = int(torch.median(w_hat_b).item())
    w_hat = max(min_w, min(max_w, w_hat))

    if w_hat < 2:
        w_hat = 2
    if L // w_hat < 2:
        w_hat = max(2, L // 2)

    return int(w_hat)


def main():
    configs = pd.read_csv(CONFIGS_CSV)
    results = []

    for ds in sorted(KEEP_DATASETS):
        rel_path, conv = DATASET_PATHS[ds]
        data = load_train_split(os.path.join(DATA, rel_path), conv)

        rows = configs[configs['dataset'] == ds]
        for _, r in rows.iterrows():
            pred_len = int(r['pred_len'])
            seq_len = int(r['seq_len'])
            best_P = int(r['period_len'])

            if len(data) < seq_len:
                continue
            n_windows = len(data) // seq_len
            windows = data[:n_windows * seq_len].reshape(n_windows, seq_len, -1)
            x = torch.from_numpy(windows).float()
            x_mean = x.mean(dim=2)

            min_w, max_w = 2, seq_len // 2
            fft_P = fft_period_length(x_mean, seq_len, min_w, max_w)

            if fft_P == best_P:
                results.append(dict(
                    dataset=ds, pred_len=pred_len, seq_len=seq_len,
                    detected_period=fft_P, matched_best_P=best_P,
                ))

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} matched-period entries to {OUT_JSON}")
    for r in results:
        print(f"  {r['dataset']:10s} pred_len={r['pred_len']:4d}  "
              f"detected_period={r['detected_period']}")


if __name__ == '__main__':
    main()
