#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_manifest(path: Path) -> list[tuple[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [(row["name"], int(row["expected_size"])) for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("scripts/bench_models_manifest.tsv"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    failed = False
    for name, expected_size in rows:
        path = args.models_dir / f"{name}.gguf"
        if not path.exists():
            print(f"MISSING\t{name}\t-\t{expected_size}")
            failed = True
            continue
        size = path.stat().st_size
        status = "OK" if size == expected_size else "BAD_SIZE"
        print(f"{status}\t{name}\t{size}\t{expected_size}")
        if size != expected_size:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
