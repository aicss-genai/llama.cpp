#!/usr/bin/env python3
"""
result_parser.py – Parse llama-bench JSON results and produce throughput statistics.

Produces:
  • Per-model prefill (pp) and decode (tg) throughput tables
  • Cross-model aggregate summary (mean ± inter-model stddev across all passing models)

Per-model statistics come directly from the JSON: avg_ts and stddev_ts are the
mean and standard deviation of the benchmark's repetitions (default r=5).

Cross-model aggregate: for each pp/tg size, mean and sample-stddev of the
per-model avg_ts values.  This is the baseline for comparing SYCL/OneDNN flags.

Usage:
    python3 scripts/result_parser.py [RESULTS_DIR]
                                     [--output FILE]
                                     [--format {text,md,csv}]
                                     [--no-per-model]

    RESULTS_DIR  Timestamped folder, e.g. results/2026-03-23-143052.
                 Defaults to the most recent folder under <repo_root>/results/.

Examples:
    python3 scripts/result_parser.py
    python3 scripts/result_parser.py results/2026-03-23-143052
    python3 scripts/result_parser.py results/2026-03-23-143052 --output baseline.md --format md
    python3 scripts/result_parser.py results/2026-03-23-143052 --output baseline.csv --format csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_latest_results_dir(repo_root: Path) -> Path:
    results_root = repo_root / "results"
    if not results_root.is_dir():
        raise FileNotFoundError(f"No results/ directory found under {repo_root}")
    dated_dirs = sorted(
        (d for d in results_root.iterdir() if d.is_dir()),
        key=lambda d: d.name,
    )
    if not dated_dirs:
        raise FileNotFoundError(f"No subdirectories found in {results_root}")
    return dated_dirs[-1]


def iter_model_jsons(results_dir: Path) -> Iterator[tuple[str, list[dict]]]:
    """Yield (model_name, entries) for every model that has a valid JSON."""
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        json_path = model_dir / f"{model_dir.name}-single-stream.json"
        if not json_path.is_file():
            print(f"  [SKIP] {model_dir.name}: no single-stream.json", file=sys.stderr)
            continue
        try:
            entries = json.loads(json_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"  [SKIP] {model_dir.name}: JSON parse error – {exc}", file=sys.stderr)
            continue
        if not entries:
            print(f"  [SKIP] {model_dir.name}: empty JSON", file=sys.stderr)
            continue
        yield model_dir.name, entries

def _extract_docker_log_reason(log_path: Path) -> str:
    """Parse a docker.log and return the most informative error summary.

    Priority order (highest wins):
      1. UR_RESULT_ERROR_*  – Level Zero / oneAPI runtime errors (e.g. OOM)
      2. SYCL error: …      – SYCL exception messages
      3. Lines containing 'error' or 'ERROR' (case-sensitive keywords)
      4. Lines containing 'exception', 'Exception', 'failed', 'Failed'
      5. The last non-empty, non-backtrace, non-address line
      6. The very last non-empty line (fallback)
    """
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return "docker.log unreadable"

    lines = [l.rstrip() for l in text.splitlines()]

    # Priority 1 – Level Zero result codes (always the most actionable)
    for l in lines:
        m = re.search(r'(UR_RESULT_ERROR_[A-Z_]+)', l)
        if m:
            return m.group(1)

    # Priority 2 – SYCL exception text (first occurrence, trim after the colon)
    for l in lines:
        if "SYCL error:" in l:
            snippet = l.split("SYCL error:")[-1].strip()
            return f"SYCL error: {snippet}"

    # Priority 3 – explicit error keywords
    _error_re = re.compile(r'\b(ERROR|error)\b')
    for l in lines:
        if _error_re.search(l) and l.strip():
            return l.strip()

    # Priority 4 – exception / failed keywords
    _exc_re = re.compile(r'\b(exception|Exception|failed|Failed)\b')
    for l in lines:
        if _exc_re.search(l) and l.strip():
            return l.strip()

    # Priority 5 – last non-empty line that doesn't look like a stack-trace frame
    _bt_re = re.compile(r'^\s*(0x[0-9a-f]+|\[0x|/[^ ]+\.so|/lib/)')
    candidates = [
        l.strip() for l in reversed(lines)
        if l.strip() and not _bt_re.match(l)
    ]
    if candidates:
        return candidates[0]

    # Fallback – absolute last line
    non_empty = [l.strip() for l in lines if l.strip()]
    return non_empty[-1] if non_empty else "docker.log empty"


def collect_skipped_models(results_dir: Path) -> list[tuple[str, str]]:
    """Return list of (dir_name, reason) for model directories without a valid JSON."""
    skipped: list[tuple[str, str]] = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        json_path = model_dir / f"{model_dir.name}-single-stream.json"
        if not json_path.is_file():
            log_path = model_dir / f"{model_dir.name}-docker.log"
            if log_path.is_file():
                reason = _extract_docker_log_reason(log_path)
            else:
                reason = "no single-stream.json (no docker.log)"
            skipped.append((model_dir.name, reason))
        else:
            try:
                entries = json.loads(json_path.read_text())
                if not entries:
                    skipped.append((model_dir.name, "empty JSON"))
            except json.JSONDecodeError as exc:
                skipped.append((model_dir.name, f"JSON parse error: {exc}"))
    return skipped

def split_pp_tg(
    entries: list[dict],
) -> tuple[dict[int, dict], dict[int, dict]]:
    """
    Return:
        pp_map: {n_prompt: entry}  where n_gen == 0
        tg_map: {n_gen:    entry}  where n_prompt == 0
    """
    pp_map: dict[int, dict] = {}
    tg_map: dict[int, dict] = {}
    for e in entries:
        if e.get("n_gen", -1) == 0 and e.get("n_prompt", 0) > 0:
            pp_map[e["n_prompt"]] = e
        elif e.get("n_prompt", -1) == 0 and e.get("n_gen", 0) > 0:
            tg_map[e["n_gen"]] = e
    return pp_map, tg_map


def extract_model_meta(entries: list[dict]) -> dict:
    """Pull top-level metadata from the first entry."""
    first = entries[0]
    return {
        "model_type": first.get("model_type", ""),
        "model_size_gib": first.get("model_size", 0) / (1024 ** 3),
        "gpu_info": first.get("gpu_info", ""),
        "backends": first.get("backends", ""),
        "build_commit": first.get("build_commit", ""),
        "flash_attn": first.get("flash_attn", False),
        "n_batch": first.get("n_batch", 0),
        "n_ubatch": first.get("n_ubatch", 0),
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def sample_stddev(values: list[float]) -> float:
    """Sample standard deviation (N-1); returns 0.0 for < 2 values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def aggregate_across_models(
    per_model: dict[str, tuple[dict[int, dict], dict[int, dict]]],
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    """
    Return aggregate (pp_agg, tg_agg) where each is:
        {size: (mean_avg_ts, stddev_of_avg_ts_across_models)}
    stddev is the inter-model sample stddev – the spread of raw throughput
    numbers across different model architectures at each prompt/gen size.
    """
    pp_values: dict[int, list[float]] = defaultdict(list)
    tg_values: dict[int, list[float]] = defaultdict(list)

    for _model_name, (pp_map, tg_map) in per_model.items():
        for n_prompt, e in pp_map.items():
            pp_values[n_prompt].append(e["avg_ts"])
        for n_gen, e in tg_map.items():
            tg_values[n_gen].append(e["avg_ts"])

    pp_agg = {
        k: (sum(v) / len(v), sample_stddev(v))
        for k, v in sorted(pp_values.items())
    }
    tg_agg = {
        k: (sum(v) / len(v), sample_stddev(v))
        for k, v in sorted(tg_values.items())
    }
    return pp_agg, tg_agg


# ---------------------------------------------------------------------------
# Model metadata helpers (used by comparison CSV)
# ---------------------------------------------------------------------------

def extract_model_name(dir_name: str) -> str:
    """Strip image-tag suffix from a results directory name.

    e.g. 'DeepSeek-R1-Qwen-32B-Q4-0.14.0-b8.1' -> 'DeepSeek-R1-Qwen-32B-Q4'
    Matches a trailing '-<major>.<minor>.<patch>-<tag>' pattern.
    """
    return re.sub(r'-\d+\.\d+\.\d+-[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*$', '', dir_name)


def normalize_quant(raw: str) -> str:
    """Normalise quant descriptor to a compact identifier.

    'Q4_K - Medium' -> 'Q4_K_M',  'Q8_0' -> 'Q8_0'
    """
    raw = raw.strip()
    raw = re.sub(r'\s*-\s*Medium', '_M', raw)
    raw = re.sub(r'\s*-\s*Small',  '_S', raw)
    raw = re.sub(r'\s*-\s*Large',  '_L', raw)
    return re.sub(r'\s+', '_', raw)


def parse_model_type(model_type_str: str, model_n_params: int) -> tuple[str, float, str]:
    """Return (arch, params_b, quant) extracted from a model_type string.

    Examples:
      'qwen2 32B Q4_K - Medium'  -> ('qwen2', 32.8, 'Q4_K_M')  # params_b from n_params
      'llama 8B Q8_0'            -> ('llama',  8.0, 'Q8_0')
      'gemma2 9B Q4_K - Medium'  -> ('gemma2', 9.0, 'Q4_K_M')
    """
    tokens = model_type_str.split()
    arch = tokens[0] if tokens else ""

    # Collect quant tokens: skip arch (index 0) and any param-size token (e.g. "32B", "3.8B")
    quant_tokens: list[str] = []
    skipped_params = False
    for tok in tokens[1:]:
        if not skipped_params and re.match(r'^\d+(?:\.\d+)?[BbMm]$', tok):
            skipped_params = True
            continue
        quant_tokens.append(tok)

    quant = normalize_quant(" ".join(quant_tokens))
    params_b = round(model_n_params / 1e9, 1) if model_n_params else 0.0
    return arch, params_b, quant


def classify_tier(params_b: float) -> str:
    """Map param count to a size tier label."""
    if params_b <= 4:
        return "small"
    if params_b <= 14:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def _col_widths(*cols: list[str]) -> list[int]:
    return [max(len(cell) for cell in col) for col in cols]


def format_throughput_table_text(
    title: str,
    size_label: str,
    rows: list[tuple[int, float, float]],  # (size, avg_ts, stddev_ts)
) -> str:
    """Plain-text table matching the layout in the spec."""
    size_col = [size_label] + [str(r[0]) for r in rows]
    avg_col = ["avg t/s"] + [f"{r[1]:.2f}" for r in rows]
    std_col = ["stddev"] + [f"±{r[2]:.2f}" for r in rows]

    sw, aw, dw = _col_widths(size_col, avg_col, std_col)

    sep = f"{'─' * (sw + 2)}{'─' * (aw + 2)}{'─' * (dw + 2)}"
    lines = [
        title,
        sep,
        f"  {size_col[0]:<{sw}}  {avg_col[0]:>{aw}}  {std_col[0]:>{dw}}",
        sep,
    ]
    for i in range(1, len(size_col)):
        lines.append(
            f"  {size_col[i]:<{sw}}  {avg_col[i]:>{aw}}  {std_col[i]:>{dw}}"
        )
    lines.append(sep)
    return "\n".join(lines)


def format_throughput_table_md(
    title: str,
    size_label: str,
    rows: list[tuple[int, float, float]],
) -> str:
    lines = [
        f"### {title}",
        "",
        f"| {size_label} | avg t/s | stddev |",
        "|---:|---:|---:|",
    ]
    for size, avg, std in rows:
        lines.append(f"| {size} | {avg:.2f} | ±{std:.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison CSV format (one row per model × task × token-size)
# ---------------------------------------------------------------------------

# Row 1: section group labels (first cell of each group; rest are empty)
_CSV_HDR1 = [
    "Configuration", "", "", "", "", "",
    "Throughput (tok/s)", "", "", "", "", "",
    "Speed Ratios", "", "",
    "VRAM (MiB)", "",
    "Analysis", "",
    "E2E Latency (ms)", "", "", "",
    "",  # Note
]

# Row 2: individual column headers (embedded newlines render as wrapped text in spreadsheets)
_CSV_HDR2 = [
    "Model", "Params (B)", "Quant", "Arch", "Task\nType", "Tokens",
    "SYCL\n(tok/s)", "SYCL\nStdDev",
    "Vulkan\n(tok/s)", "Vulkan\nStdDev",
    "CUDA\n(tok/s)", "CUDA\nStdDev",
    "CUDA/\nSYCL", "CUDA/\nVulkan", "SYCL/\nVulkan",
    "SYCL\nVRAM", "CUDA\nVRAM",
    "Fastest\nBackend", "Tier",
    "SYCL E2E (ms)", "Vulkan E2E (ms)", "CUDA E2E (ms)", "CUDA/SYCL Ratio",
    "Note",
]


def write_comparison_csv(
    per_model: dict[str, tuple[dict, dict, list[dict]]],
    out_file: io.TextIOBase,
    skipped_models: list[tuple[str, str]] | None = None,
) -> None:
    """Write one row per (model × task-type × token-size) in comparison-spreadsheet format.

    Row order: all pp rows across all models, then all tg rows across all models.
    SYCL columns are populated from the JSON results.
    Vulkan, CUDA, VRAM, and cross-backend ratio columns are left empty so the
    spreadsheet can be filled in when those backends are benchmarked.
    E2E latency (ms) = tokens / avg_ts * 1000.
    Skipped/errored models are appended at the end with an error note.
    """
    writer = csv.writer(out_file)
    writer.writerow(_CSV_HDR1)
    writer.writerow(_CSV_HDR2)

    # Collect pp and tg rows separately so we can emit all pp first, then all tg
    pp_rows_out: list[list] = []
    tg_rows_out: list[list] = []

    for model_dir_name, (pp_map, tg_map, entries) in per_model.items():
        if not entries:
            continue
        first = entries[0]
        model_name = extract_model_name(model_dir_name)
        arch, params_b, quant = parse_model_type(
            first.get("model_type", ""),
            first.get("model_n_params", 0),
        )
        tier = classify_tier(params_b)

        for n_prompt, e in sorted(pp_map.items()):
            avg_ts = e["avg_ts"]
            stddev = e["stddev_ts"]
            e2e_sycl = f"{n_prompt / avg_ts * 1000:.3f}" if avg_ts else ""
            pp_rows_out.append([
                model_name, f"{params_b:.1f}", quant, arch, "pp", n_prompt,
                f"{avg_ts:.2f}", f"{stddev:.2f}",
                "", "",   # Vulkan tok/s, StdDev
                "", "",   # CUDA tok/s, StdDev
                "", "", "",  # CUDA/SYCL, CUDA/Vulkan, SYCL/Vulkan ratios
                "", "",   # SYCL VRAM, CUDA VRAM
                "SYCL", tier,
                e2e_sycl, "", "", "",  # SYCL E2E, Vulkan E2E, CUDA E2E, CUDA/SYCL ratio
                "",  # Note
            ])

        for n_gen, e in sorted(tg_map.items()):
            avg_ts = e["avg_ts"]
            stddev = e["stddev_ts"]
            e2e_sycl = f"{n_gen / avg_ts * 1000:.3f}" if avg_ts else ""
            tg_rows_out.append([
                model_name, f"{params_b:.1f}", quant, arch, "tg", n_gen,
                f"{avg_ts:.2f}", f"{stddev:.2f}",
                "", "",   # Vulkan tok/s, StdDev
                "", "",   # CUDA tok/s, StdDev
                "", "", "",  # CUDA/SYCL, CUDA/Vulkan, SYCL/Vulkan ratios
                "", "",   # SYCL VRAM, CUDA VRAM
                "SYCL", tier,
                e2e_sycl, "", "", "",  # SYCL E2E, Vulkan E2E, CUDA E2E, CUDA/SYCL ratio
                "",  # Note
            ])

    for row in pp_rows_out:
        writer.writerow(row)
    for row in tg_rows_out:
        writer.writerow(row)

    # Skipped / errored models – one stub row per model with the Note column populated
    for dir_name, reason in (skipped_models or []):
        model_name = extract_model_name(dir_name)
        writer.writerow([
            model_name, "", "", "", "N/A", "",
            "", "", "", "", "", "",
            "", "", "", "", "", "SKIPPED", "",
            "", "", "", "",
            reason,
        ])


# ---------------------------------------------------------------------------
# Per-model report
# ---------------------------------------------------------------------------

def report_model(
    model_name: str,
    entries: list[dict],
    fmt: str,
    out: io.TextIOBase,
) -> None:
    meta = extract_model_meta(entries)
    pp_map, tg_map = split_pp_tg(entries)

    pp_rows = [
        (n, e["avg_ts"], e["stddev_ts"])
        for n, e in sorted(pp_map.items())
    ]
    tg_rows = [
        (n, e["avg_ts"], e["stddev_ts"])
        for n, e in sorted(tg_map.items())
    ]

    if fmt == "text":
        out.write(f"\n{'═' * 72}\n")
        out.write(f"  Model : {model_name}\n")
        out.write(f"  Type  : {meta['model_type']}   ({meta['model_size_gib']:.2f} GiB)\n")
        out.write(f"  GPU   : {meta['gpu_info']}   backends={meta['backends']}\n")
        out.write(f"  Build : {meta['build_commit']}   fa={meta['flash_attn']}   "
                  f"n_batch={meta['n_batch']}\n")
        out.write(f"{'═' * 72}\n\n")

        if pp_rows:
            out.write(
                format_throughput_table_text(
                    "Prefill throughput (pp, n_gen=0) — avg t/s ± stddev",
                    "n_prompt",
                    pp_rows,
                )
            )
            out.write("\n\n")
        if tg_rows:
            out.write(
                format_throughput_table_text(
                    "Decode throughput (tg, n_prompt=0) — avg t/s ± stddev",
                    "n_gen",
                    tg_rows,
                )
            )
            out.write("\n\n")

    elif fmt == "md":
        out.write(f"\n## {model_name}\n\n")
        out.write(
            f"| Field | Value |\n|---|---|\n"
            f"| model_type | {meta['model_type']} |\n"
            f"| size_gib | {meta['model_size_gib']:.2f} |\n"
            f"| gpu_info | {meta['gpu_info']} |\n"
            f"| backends | {meta['backends']} |\n"
            f"| build_commit | {meta['build_commit']} |\n"
            f"| flash_attn | {meta['flash_attn']} |\n"
            f"| n_batch | {meta['n_batch']} |\n\n"
        )
        if pp_rows:
            out.write(
                format_throughput_table_md(
                    "Prefill throughput (pp, n_gen=0) — avg t/s ± stddev",
                    "n_prompt",
                    pp_rows,
                )
            )
            out.write("\n\n")
        if tg_rows:
            out.write(
                format_throughput_table_md(
                    "Decode throughput (tg, n_prompt=0) — avg t/s ± stddev",
                    "n_gen",
                    tg_rows,
                )
            )
            out.write("\n\n")





# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def report_aggregate(
    results_dir: Path,
    n_models: int,
    pp_agg: dict[int, tuple[float, float]],
    tg_agg: dict[int, tuple[float, float]],
    fmt: str,
    out: io.TextIOBase,
) -> None:
    pp_rows = [(k, mean, std) for k, (mean, std) in sorted(pp_agg.items())]
    tg_rows = [(k, mean, std) for k, (mean, std) in sorted(tg_agg.items())]

    if fmt == "text":
        out.write(f"\n{'═' * 72}\n")
        out.write(f"  AGGREGATE — {n_models} model(s) — {results_dir.name}\n")
        out.write("  (mean ± inter-model sample-stddev of per-model avg_ts)\n")
        out.write(f"{'═' * 72}\n\n")

        if pp_rows:
            out.write(
                format_throughput_table_text(
                    "Prefill throughput (pp, n_gen=0) — avg t/s ± stddev",
                    "n_prompt",
                    pp_rows,
                )
            )
            out.write("\n\n")
        if tg_rows:
            out.write(
                format_throughput_table_text(
                    "Decode throughput (tg, n_prompt=0) — avg t/s ± stddev",
                    "n_gen",
                    tg_rows,
                )
            )
            out.write("\n\n")

    elif fmt == "md":
        out.write(f"\n---\n\n## AGGREGATE — {n_models} model(s) — {results_dir.name}\n\n")
        out.write(
            "> mean and inter-model sample-stddev of per-model avg\\_ts values.\n"
            "> stddev reflects spread across different model architectures at each size.\n\n"
        )
        if pp_rows:
            out.write(
                format_throughput_table_md(
                    "Prefill throughput (pp, n_gen=0) — avg t/s ± stddev",
                    "n_prompt",
                    pp_rows,
                )
            )
            out.write("\n\n")
        if tg_rows:
            out.write(
                format_throughput_table_md(
                    "Decode throughput (tg, n_prompt=0) — avg t/s ± stddev",
                    "n_gen",
                    tg_rows,
                )
            )
            out.write("\n\n")





# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Path to a dated results directory (e.g. results/2026-03-23). "
             "Defaults to the most recent folder under <repo_root>/results/.",
    )
    p.add_argument(
        "--output", "-o",
        metavar="FILE",
        default=None,
        help="Write report to FILE instead of stdout. "
             "Format is inferred from the file extension (.md / .csv) "
             "unless --format is also specified.",
    )
    p.add_argument(
        "--format", "-f",
        choices=["text", "md", "csv"],
        default=None,
        help="Output format: text (default for stdout), md, or csv.",
    )
    p.add_argument(
        "--no-per-model",
        action="store_true",
        default=False,
        help="Skip per-model tables; print only the aggregate summary.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Resolve results directory
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = Path.cwd() / results_dir
    else:
        results_dir = find_latest_results_dir(repo_root)

    if not results_dir.is_dir():
        sys.exit(f"ERROR: results directory not found: {results_dir}")

    # Resolve output format
    fmt = args.format
    if fmt is None:
        if args.output:
            ext = Path(args.output).suffix.lower()
            fmt = {"md": "md", ".md": "md", ".csv": "csv"}.get(ext, "text")
        else:
            fmt = "text"

    print(f"Parsing results in: {results_dir}", file=sys.stderr)

    # Collect skipped/errored models before loading successes, then load passing models.
    # iter_model_jsons also prints [SKIP] lines; suppress duplicate stderr noise here
    # by running collect_skipped_models first and letting iter_model_jsons stay silent.
    skipped_models = collect_skipped_models(results_dir)
    if skipped_models:
        for dir_name, reason in skipped_models:
            print(f"  [SKIP] {dir_name}: {reason}", file=sys.stderr)

    # Load all models (skip-reason already printed above via collect_skipped_models;
    # redirect stderr during iter_model_jsons to suppress duplicate [SKIP] lines)
    per_model: dict[str, tuple[dict, dict]] = {}
    import io as _io
    _null_err = _io.StringIO()
    _real_stderr, sys.stderr = sys.stderr, _null_err
    try:
        for model_name, entries in iter_model_jsons(results_dir):
            pp_map, tg_map = split_pp_tg(entries)
            per_model[model_name] = (pp_map, tg_map, entries)
    finally:
        sys.stderr = _real_stderr
    del _io, _null_err, _real_stderr

    if not per_model:
        sys.exit("ERROR: no valid result files found.")

    print(f"Loaded {len(per_model)} model(s).", file=sys.stderr)

    # Compute aggregate across models
    model_pp_tg = {name: (val[0], val[1]) for name, val in per_model.items()}
    pp_agg, tg_agg = aggregate_across_models(model_pp_tg)

    # Open output
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", encoding="utf-8", newline="")
    else:
        out_file = sys.stdout

    try:
        if fmt == "csv":
            write_comparison_csv(per_model, out_file, skipped_models)

        else:
            if fmt == "md":
                out_file.write(f"# llama-bench Throughput Report — {results_dir.name}\n\n")
                out_file.write(
                    "_Generated by `scripts/result_parser.py`. "
                    "Use as baseline for SYCL/OneDNN optimization comparisons._\n\n"
                )

            if not args.no_per_model:
                for model_name, (pp_map, tg_map, entries) in per_model.items():
                    report_model(model_name, entries, fmt, out_file)

            report_aggregate(results_dir, len(per_model), pp_agg, tg_agg, fmt, out_file)

    finally:
        if args.output:
            out_file.close()
            print(f"Report written to: {out_file.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
