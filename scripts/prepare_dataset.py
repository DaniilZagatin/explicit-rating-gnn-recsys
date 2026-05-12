from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    args = parser.parse_args()

    ratings = pd.read_csv(args.ratings)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "timestamp" in ratings.columns:
        ratings = ratings.sort_values(["userId", "timestamp"])
    ratings["is_positive"] = ratings["rating"] >= args.positive_threshold
    ratings.to_parquet(output_dir / "interactions.parquet", index=False)
    print(output_dir / "interactions.parquet")


if __name__ == "__main__":
    main()
