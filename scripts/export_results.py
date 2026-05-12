from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/leaderboard.md")
    args = parser.parse_args()

    source = Path(args.input)
    if source.suffix == ".csv":
        table = pd.read_csv(source)
    elif source.suffix in {".parquet", ".pq"}:
        table = pd.read_parquet(source)
    else:
        raise ValueError("Use CSV or Parquet as input")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table.to_markdown(index=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
