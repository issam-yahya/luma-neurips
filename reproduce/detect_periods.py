"""
Offline FFT-based period detection for LUMA.

For each dataset–horizon configuration, the script:

1. loads only the training split;
2. extracts eligible input windows;
3. averages each window across variables;
4. removes the FFT DC component;
5. detects the dominant valid period per window;
6. reports the median detected period.

Output:
    reproduce/detected_periods.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_DIR = REPO_ROOT / "dataset"
DEFAULT_CONFIGS = SCRIPT_DIR / "configs_luma_best.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "detected_periods.json"


DATASET_PATHS = {
    "ETTh1": ("ETT-small/ETTh1.csv", "ETTh"),
    "ETTh2": ("ETT-small/ETTh2.csv", "ETTh"),
    "ETTm1": ("ETT-small/ETTm1.csv", "ETTm"),
    "ETTm2": ("ETT-small/ETTm2.csv", "ETTm"),
    "weather": ("weather/weather.csv", "ratio"),
    "electricity": ("electricity/electricity.csv", "ratio"),
    "traffic": ("traffic/traffic.csv", "ratio"),
    "exchange": ("exchange_rate/exchange_rate.csv", "ratio"),
    "ili": ("illness/national_illness.csv", "ratio"),
}


def load_train_split(path: Path, convention: str) -> np.ndarray:
    """Load only the training partition."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    feature_columns = [column for column in df.columns if column != "date"]

    if not feature_columns:
        raise ValueError(f"No feature columns found in {path}")

    data = df[feature_columns].to_numpy(dtype=np.float32)

    if convention == "ETTh":
        train_end = 12 * 30 * 24
    elif convention == "ETTm":
        train_end = 12 * 30 * 24 * 4
    elif convention == "ratio":
        train_end = int(len(data) * 0.7)
    else:
        raise ValueError(f"Unknown split convention: {convention}")

    return data[:train_end]


def detect_periods_per_window(
    windows: torch.Tensor,
    seq_len: int,
    min_period: int,
    max_period: int,
) -> torch.Tensor:
    """
    Detect the dominant FFT period for each window.

    Args:
        windows: Tensor of shape [N, seq_len].
    """
    spectrum = torch.fft.rfft(windows, dim=1)
    power = spectrum.real.square() + spectrum.imag.square()

    # Remove the zero-frequency/DC component.
    power[:, 0] = 0.0

    frequencies = torch.arange(
        power.shape[1],
        device=power.device,
    )

    candidate_periods = torch.where(
        frequencies > 0,
        (
            seq_len / frequencies.clamp(min=1)
        ).round().long(),
        torch.zeros_like(frequencies),
    )

    valid = (
        (candidate_periods >= min_period)
        & (candidate_periods <= max_period)
    )

    if not bool(valid.any()):
        raise ValueError(
            f"No valid frequency bins for seq_len={seq_len}, "
            f"period range=[{min_period}, {max_period}]"
        )

    masked_power = power.masked_fill(
        ~valid.unsqueeze(0),
        float("-inf"),
    )

    dominant_bins = masked_power.argmax(dim=1)

    detected_periods = (
        seq_len / dominant_bins.clamp(min=1)
    ).round().long()

    return detected_periods.clamp(
        min=min_period,
        max=max_period,
    )


def detect_dataset_period(
    train_data: np.ndarray,
    seq_len: int,
    pred_len: int,
    min_period: int,
    max_period: int,
    stride: int,
    batch_size: int,
) -> tuple[int, int]:
    """Return the median period across eligible training windows."""
    eligible_windows = len(train_data) - seq_len - pred_len + 1

    if eligible_windows <= 0:
        raise ValueError(
            f"Training split too short for seq_len={seq_len} "
            f"and pred_len={pred_len}"
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        train_data,
        window_shape=seq_len,
        axis=0,
    )

    # sliding_window_view returns [N, C, L].
    windows = windows[:eligible_windows:stride]
    windows = windows.mean(axis=1)

    detected_batches = []

    for start in range(0, len(windows), batch_size):
        batch_array = np.ascontiguousarray(
            windows[start : start + batch_size]
        )
        batch = torch.from_numpy(batch_array).float()

        periods = detect_periods_per_window(
            windows=batch,
            seq_len=seq_len,
            min_period=min_period,
            max_period=max_period,
        )

        detected_batches.append(periods.cpu())

    all_periods = torch.cat(detected_batches)

    return (
        int(torch.median(all_periods).item()),
        int(all_periods.numel()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect LUMA periods using training-set FFT."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=DEFAULT_CONFIGS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--min-period",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-period",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset subset, e.g. ETTh1 traffic",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.configs.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {args.configs}"
        )

    configs = pd.read_csv(args.configs)

    required_columns = {
        "dataset",
        "pred_len",
        "seq_len",
    }

    missing = required_columns.difference(configs.columns)

    if missing:
        raise ValueError(
            "Missing configuration columns: "
            + ", ".join(sorted(missing))
        )

    if args.datasets:
        configs = configs[
            configs["dataset"].isin(args.datasets)
        ]

    configs = (
        configs[
            ["dataset", "pred_len", "seq_len"]
        ]
        .drop_duplicates()
        .sort_values(["dataset", "pred_len"])
    )

    train_cache: dict[str, np.ndarray] = {}
    results = []

    for row in configs.itertuples(index=False):
        dataset = str(row.dataset)
        pred_len = int(row.pred_len)
        seq_len = int(row.seq_len)

        if dataset not in DATASET_PATHS:
            print(f"Skipping unsupported dataset: {dataset}")
            continue

        relative_path, convention = DATASET_PATHS[dataset]

        if dataset not in train_cache:
            train_cache[dataset] = load_train_split(
                args.data_dir / relative_path,
                convention,
            )

        max_period = min(
            args.max_period,
            seq_len // 2,
        )

        detected_period, number_of_windows = detect_dataset_period(
            train_data=train_cache[dataset],
            seq_len=seq_len,
            pred_len=pred_len,
            min_period=args.min_period,
            max_period=max_period,
            stride=args.stride,
            batch_size=args.batch_size,
        )

        results.append(
            {
                "dataset": dataset,
                "pred_len": pred_len,
                "seq_len": seq_len,
                "detected_period": detected_period,
                "number_of_training_windows": number_of_windows,
            }
        )

        print(
            f"{dataset:12s} "
            f"H={pred_len:4d} "
            f"L={seq_len:4d} "
            f"P={detected_period:3d}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print(
        f"\nWrote {len(results)} entries to {args.output}"
    )


if __name__ == "__main__":
    main()
