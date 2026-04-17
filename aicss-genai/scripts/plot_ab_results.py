#!/usr/bin/env python3
"""Render aggregate throughput comparison charts for benchmark results runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd


DEFAULT_TITLE = "Benchmark Comparison"
DEFAULT_SUBTITLE = "Aggregate mean throughput across models; error bars show inter-model stddev"


class ComparisonInput(NamedTuple):
    label: str
    csv_path: Path


def normalize_column_name(name: str) -> str:
    collapsed = " ".join(str(name).replace("\n", " ").split())
    return collapsed.replace("/", " ").replace("(", "").replace(")", "")


def expected_columns() -> set[str]:
    return {
        "Model",
        "Task Type",
        "Tokens",
        "SYCL tok s",
        "SYCL StdDev",
    }


def csv_has_expected_columns(csv_path: Path) -> bool:
    try:
        header_df = pd.read_csv(csv_path, header=1, nrows=0)
    except Exception:
        return False

    normalized = {normalize_column_name(column) for column in header_df.columns}
    return expected_columns().issubset(normalized)


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=1)
    df = df.rename(columns={column: normalize_column_name(column) for column in df.columns})

    required_columns = expected_columns()
    missing = required_columns - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{csv_path} is missing expected columns: {missing_list}")

    return df[["Model", "Task Type", "Tokens", "SYCL tok s", "SYCL StdDev"]].copy()


def parse_labeled_input(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        # Explicit LABEL=PATH format
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"Missing label in input: {spec}")
        if not raw_path:
            raise argparse.ArgumentTypeError(f"Missing path in input: {spec}")
        return label, Path(raw_path).expanduser()
    else:
        # Path only; use last directory component as label
        raw_path = spec.strip()
        if not raw_path:
            raise argparse.ArgumentTypeError(f"Missing path in input: {spec}")
        path = Path(raw_path).expanduser()
        label = path.name
        return label, path


def resolve_csv_path(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist as a directory: {path}")

    preferred_csv = path / f"{path.name}.csv"
    if preferred_csv.is_file() and csv_has_expected_columns(preferred_csv):
        return preferred_csv

    csv_files = sorted(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {path}")

    valid_csv_files = [csv_file for csv_file in csv_files if csv_has_expected_columns(csv_file)]
    if len(valid_csv_files) == 1:
        return valid_csv_files[0]
    if len(valid_csv_files) > 1:
        matches = ", ".join(csv_file.name for csv_file in valid_csv_files)
        raise FileNotFoundError(
            f"Multiple benchmark CSV files found in directory {path}; "
            f"please remove extras or pass a directory with one benchmark CSV: {matches}"
        )

    matches = ", ".join(csv_file.name for csv_file in csv_files)
    raise FileNotFoundError(
        f"No benchmark CSV with expected columns found in directory {path}. "
        f"Found CSV files: {matches}"
    )


def load_comparison_input(spec: str) -> ComparisonInput:
    label, path = parse_labeled_input(spec)
    return ComparisonInput(label=label, csv_path=resolve_csv_path(path))


def aggregate_by_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    task_df = df[df["Task Type"] == task].copy()
    task_df["Tokens"] = pd.to_numeric(task_df["Tokens"], errors="raise")
    task_df["SYCL tok s"] = pd.to_numeric(task_df["SYCL tok s"], errors="raise")

    summary = (
        task_df.groupby("Tokens", as_index=False)
        .agg(
            mean_throughput=("SYCL tok s", "mean"),
            inter_model_stddev=("SYCL tok s", lambda values: values.std(ddof=1)),
            model_count=("Model", "nunique"),
        )
        .sort_values("Tokens")
    )
    summary["inter_model_stddev"] = summary["inter_model_stddev"].fillna(0.0)
    return summary


def build_comparison(first_df: pd.DataFrame, second_df: pd.DataFrame, task: str) -> pd.DataFrame:
    baseline = aggregate_by_task(first_df, task).rename(
        columns={
            "mean_throughput": "baseline_mean",
            "inter_model_stddev": "baseline_stddev",
            "model_count": "baseline_models",
        }
    )
    graphs = aggregate_by_task(second_df, task).rename(
        columns={
            "mean_throughput": "graphs_mean",
            "inter_model_stddev": "graphs_stddev",
            "model_count": "graphs_models",
        }
    )
    merged = baseline.merge(graphs, on="Tokens", how="inner")
    merged["delta_pct"] = ((merged["graphs_mean"] / merged["baseline_mean"]) - 1.0) * 100.0
    return merged


def add_value_labels(ax: Axes, bars, values: pd.Series, y_limit: float) -> None:
    for bar, value in zip(bars, values, strict=True):
        y_offset = max(value * 0.012, y_limit * 0.01)
        x_offset = bar.get_width() * 0.18
        ax.text(
            bar.get_x() + bar.get_width() / 2 - x_offset,
            value + y_offset,
            f"{value:.1f}",
            ha="right",
            va="bottom",
            fontsize=9,
            rotation=90,
            color="black",
        )


def annotate_deltas(ax: Axes, x_positions: np.ndarray, comparison: pd.DataFrame) -> None:
    for xpos, baseline_mean, graphs_mean, baseline_stddev, graphs_stddev, delta_pct in zip(
        x_positions,
        comparison["baseline_mean"],
        comparison["graphs_mean"],
        comparison["baseline_stddev"],
        comparison["graphs_stddev"],
        comparison["delta_pct"],
        strict=True,
    ):
        ymax = max(baseline_mean + baseline_stddev, graphs_mean + graphs_stddev)
        ax.text(
            xpos,
            ymax * 1.03,
            f"{delta_pct:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def plot_comparison(
    prefill: pd.DataFrame,
    decode: pd.DataFrame,
    output_path: Path,
    title: str,
    baseline_label: str,
    graphs_label: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), constrained_layout=True)

    colors = {
        "baseline": "#5b7184",
        "graphs": "#d06b37",
    }
    tasks = [
        (axes[0], prefill, "Prompt Processing", "Prompt tokens"),
        (axes[1], decode, "Text Generation", "Generated tokens"),
    ]
    bar_width = 0.36

    for ax, comparison, panel_title, xlabel in tasks:
        x_positions = np.arange(len(comparison))
        max_height = max(
            (comparison["baseline_mean"] + comparison["baseline_stddev"]).max(),
            (comparison["graphs_mean"] + comparison["graphs_stddev"]).max(),
        )
        baseline_bars = ax.bar(
            x_positions - bar_width / 2,
            comparison["baseline_mean"],
            bar_width,
            yerr=comparison["baseline_stddev"],
            capsize=5,
            color=colors["baseline"],
            label=baseline_label,
            alpha=0.95,
        )
        graphs_bars = ax.bar(
            x_positions + bar_width / 2,
            comparison["graphs_mean"],
            bar_width,
            yerr=comparison["graphs_stddev"],
            capsize=5,
            color=colors["graphs"],
            label=graphs_label,
            alpha=0.95,
        )

        ax.set_title(panel_title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xticks(x_positions, [str(token) for token in comparison["Tokens"]])
        ax.legend(frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        add_value_labels(
            ax,
            baseline_bars,
            comparison["baseline_mean"],
            max_height,
        )
        add_value_labels(
            ax,
            graphs_bars,
            comparison["graphs_mean"],
            max_height,
        )
        annotate_deltas(ax, x_positions, comparison)

        ax.set_ylim(0, max_height * 1.22)

    combined_title = f"{title}\n{DEFAULT_SUBTITLE}"
    fig.suptitle(combined_title, fontsize=15, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def print_summary(
    prefill: pd.DataFrame,
    decode: pd.DataFrame,
    baseline_label: str,
    graphs_label: str,
) -> None:
    for label, comparison in (("Prefill", prefill), ("Decode", decode)):
        print(label)
        print(f"Tokens | {baseline_label} | {graphs_label} | Delta")
        for row in comparison.itertuples(index=False):
            print(
                f"{row.Tokens:>6} | "
                f"{row.baseline_mean:>8.2f} ± {row.baseline_stddev:<8.2f} | "
                f"{row.graphs_mean:>8.2f} ± {row.graphs_stddev:<8.2f} | "
                f"{row.delta_pct:+6.2f}%"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Chart title",
    )
    parser.add_argument(
        "--a",
        dest="input_a",
        type=load_comparison_input,
        required=True,
        help='First dataset as [LABEL=]PATH, where PATH is a results directory. Example: /path/to/results or "Custom Label"=/path/to/results. If LABEL is omitted, the directory name is used.',
    )
    parser.add_argument(
        "--b",
        dest="input_b",
        type=load_comparison_input,
        required=True,
        help='Second dataset as [LABEL=]PATH, where PATH is a results directory. Example: /path/to/results or "Custom Label"=/path/to/results. If LABEL is omitted, the directory name is used.',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("ab-comparison.svg"),
        help="Output path for the rendered SVG",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated list of models to exclude from aggregation",
    )
    args = parser.parse_args()

    input_a = args.input_a
    input_b = args.input_b

    dataset_a = load_results(input_a.csv_path)
    dataset_b = load_results(input_b.csv_path)

    # Filter out excluded models
    if args.exclude:
        excluded_models = {model.strip() for model in args.exclude.split(",")}
        dataset_a = dataset_a[~dataset_a["Model"].isin(excluded_models)].copy()
        dataset_b = dataset_b[~dataset_b["Model"].isin(excluded_models)].copy()

    prefill = build_comparison(dataset_a, dataset_b, "pp")
    decode = build_comparison(dataset_a, dataset_b, "tg")

    print_summary(prefill, decode, input_a.label, input_b.label)
    plot_comparison(prefill, decode, args.output, args.title, input_a.label, input_b.label)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()