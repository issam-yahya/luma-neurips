"""Calculate MACs and parameters for Electricity at prediction length 720."""

import os

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'efficiency_results.csv')

DATASET = 'electricity'
PRED_LEN = 720
PERIOD_LEN = 24
RANK = 10
SEQ_LEN = 720
ENC_IN = 321


def main():
    conv1d_macs = SEQ_LEN * ENC_IN * (PERIOD_LEN + 1)
    low_rank_macs = SEQ_LEN * ENC_IN * 2 * RANK
    macs = conv1d_macs + low_rank_macs

    conv1d_params = PERIOD_LEN + 1
    input_windows = SEQ_LEN // PERIOD_LEN
    output_windows = PRED_LEN // PERIOD_LEN
    low_rank_params = RANK * (input_windows + output_windows + 1)
    params = conv1d_params + low_rank_params

    result = pd.DataFrame([
        {
            'dataset': DATASET,
            'pred_len': PRED_LEN,
            'period_len': PERIOD_LEN,
            'rank': RANK,
            'seq_len': SEQ_LEN,
            'enc_in': ENC_IN,
            'macs': macs,
            'thop_params': params,
            'true_trainable_params': params,
        }
    ])
    result.to_csv(OUTPUT_CSV, index=False)
    print(result.to_string(index=False))
    print(f'\nSaved to {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
