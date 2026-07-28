"""
Verify a reproduction run against the reported reference numbers.

Reads:
  reproduce/configs_luma_best.csv          (reference mse/mae per config)
  logs_reproduce/reproduced_results.csv    (freshly reproduced mse/mae)

Prints a per-config comparison and an overall PASS/FAIL. A config passes if the
reproduced MSE is within --tol relative tolerance of the reference (default 2%),
which absorbs the small run-to-run variation from GPU nondeterminism while still
catching real regressions.

Usage:
  python reproduce/check_reproduction.py
  python reproduce/check_reproduction.py --tol 0.03
"""
import argparse
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tol', type=float, default=0.02,
                    help='relative MSE tolerance (default 0.02 = 2%%)')
    ap.add_argument('--reproduced', default=os.path.join(
        REPO, 'logs_reproduce', 'reproduced_results.csv'))
    ap.add_argument('--reference', default=os.path.join(
        HERE, 'configs_luma_best.csv'))
    args = ap.parse_args()

    if not os.path.exists(args.reproduced):
        sys.exit(f'ERROR: {args.reproduced} not found -- run reproduce_luma.sh first')

    ref = pd.read_csv(args.reference)
    rep = pd.read_csv(args.reproduced)

    key = ['dataset', 'pred_len']
    ref = ref[key + ['period_len', 'rank', 'ref_mse', 'ref_mae']]
    rep = rep[key + ['mse', 'mae']].copy()
    for c in ['mse', 'mae']:
        rep[c] = pd.to_numeric(rep[c], errors='coerce')

    m = ref.merge(rep, on=key, how='left').sort_values(key)
    m['mse_rel_%'] = 100.0 * (m['mse'] - m['ref_mse']) / m['ref_mse']
    m['status'] = m.apply(
        lambda r: 'MISSING' if pd.isna(r['mse'])
        else ('PASS' if abs(r['mse'] - r['ref_mse']) <= args.tol * r['ref_mse']
              else 'FAIL'),
        axis=1)

    pd.set_option('display.width', 160)
    show = m[['dataset', 'pred_len', 'period_len', 'rank',
              'ref_mse', 'mse', 'mse_rel_%', 'status']].copy()
    show.columns = ['dataset', 'pred', 'P', 'R',
                    'ref_mse', 'repro_mse', 'rel_%', 'status']
    print(show.round(6).to_string(index=False))

    n = len(m)
    n_pass = (m['status'] == 'PASS').sum()
    n_fail = (m['status'] == 'FAIL').sum()
    n_miss = (m['status'] == 'MISSING').sum()
    worst = m.loc[m['mse_rel_%'].abs().idxmax()] if m['mse'].notna().any() else None

    print(f'\ntolerance: {args.tol*100:.1f}%   '
          f'PASS {n_pass}/{n}   FAIL {n_fail}   MISSING {n_miss}')
    if worst is not None:
        print(f'largest deviation: {worst["dataset"]} pred={int(worst["pred_len"])} '
              f'{worst["mse_rel_%"]:+.2f}%')

    if n_fail == 0 and n_miss == 0:
        print('\nRESULT: PASS -- all configs reproduce within tolerance.')
        sys.exit(0)
    else:
        print('\nRESULT: FAIL -- see FAIL/MISSING rows above.')
        sys.exit(1)


if __name__ == '__main__':
    main()
