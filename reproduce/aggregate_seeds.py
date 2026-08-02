"""Aggregate multi_seed_runs.csv into aggregated.csv."""

import argparse
import os

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS_RUNS_DIR = os.path.join(SCRIPT_DIR, 'seeds_runs')
INPUT_CSV = os.path.join(SEEDS_RUNS_DIR, 'multi_seed_runs.csv')
OUTPUT_CSV = os.path.join(SEEDS_RUNS_DIR, 'aggregated.csv')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=INPUT_CSV)
    parser.add_argument('--output', default=OUTPUT_CSV)
    return parser.parse_args()


def main():
    args = parse_args()
    results = pd.read_csv(args.input)
    results['mse'] = pd.to_numeric(results['mse'], errors='coerce')
    results = results.dropna(subset=['mse'])

    aggregated = (
        results.groupby(['dataset', 'pred_len'])
        .agg(
            n_seeds=('seed', 'count'),
            mse_mean=('mse', 'mean'),
            mse_std=('mse', 'std'),
        )
        .reset_index()
        [['dataset', 'pred_len', 'n_seeds', 'mse_mean', 'mse_std']]
    )
    aggregated.to_csv(args.output, index=False)
    print(aggregated.to_string(index=False))
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
