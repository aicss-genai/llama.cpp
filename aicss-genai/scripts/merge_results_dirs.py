#!/usr/bin/env python3
"""Merge multiple timestamped results directories into one new directory.

This utility copies model subdirectories from multiple benchmark runs into a
single output directory. If a model subdirectory name appears in more than one
input run, the merged directory name is disambiguated by appending the source
run name.

Example:
    python3 scripts/merge_results_dirs.py \
        results/2026-03-30-190830 \
        results/2026-04-01-182150 \
        --output results/merged_2026-04-02

Collision behavior:
  - first occurrence: <model-dir>
  - later occurrences: <model-dir>__<run-dir>

When a directory is renamed during merge, top-level files inside that model
directory that start with '<old-name>-' are also renamed to '<new-name>-'. This
keeps compatibility with scripts that derive file names from directory names.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input_dirs",
        nargs="+",
        help="One or more existing results run directories to merge.",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path for the merged output directory (must not already exist unless --force).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without creating files.",
    )
    return p


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _rename_prefixed_files(model_dir: Path, old_name: str, new_name: str) -> None:
    old_prefix = f"{old_name}-"
    new_prefix = f"{new_name}-"
    for child in model_dir.iterdir():
        if not child.is_file():
            continue
        if child.name.startswith(old_prefix):
            child.rename(model_dir / f"{new_prefix}{child.name[len(old_prefix):]}")


def _next_available_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        return base_name
    i = 2
    while True:
        candidate = f"{base_name}__{i}"
        if candidate not in used_names:
            return candidate
        i += 1


def main() -> None:
    args = build_parser().parse_args()

    input_dirs = [_resolve_path(d) for d in args.input_dirs]
    output_dir = _resolve_path(args.output)

    for d in input_dirs:
        if not d.is_dir():
            sys.exit(f"ERROR: input directory not found: {d}")

    if output_dir.exists():
        if args.force and not args.dry_run:
            shutil.rmtree(output_dir)
        else:
            sys.exit(
                f"ERROR: output directory already exists: {output_dir} "
                "(use --force to overwrite)"
            )

    print(f"Merging {len(input_dirs)} run(s) into: {output_dir}", file=sys.stderr)

    used_model_dirs: set[str] = set()
    manifest_rows: list[list[str]] = []

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=False)

    for run_dir in input_dirs:
        run_name = run_dir.name
        print(f"- scanning {run_dir}", file=sys.stderr)

        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue

            src_model_name = child.name
            dest_model_name = src_model_name
            collision = "no"

            if dest_model_name in used_model_dirs:
                collision = "yes"
                dest_model_name = _next_available_name(
                    f"{src_model_name}__{run_name}",
                    used_model_dirs,
                )

            used_model_dirs.add(dest_model_name)

            src_path = child
            dest_path = output_dir / dest_model_name

            manifest_rows.append([
                run_name,
                src_model_name,
                dest_model_name,
                collision,
            ])

            if args.dry_run:
                print(f"  [DRY-RUN] {src_path} -> {dest_path}", file=sys.stderr)
                continue

            shutil.copytree(src_path, dest_path)
            if dest_model_name != src_model_name:
                _rename_prefixed_files(dest_path, src_model_name, dest_model_name)

    if args.dry_run:
        print("Dry run complete. No files were written.", file=sys.stderr)
        return

    manifest_path = output_dir / "merge_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "source_model_dir", "merged_model_dir", "name_collision"])
        writer.writerows(manifest_rows)

    print(f"Merge complete. Created {len(manifest_rows)} model directory copy/copies.", file=sys.stderr)
    print(f"Manifest: {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
