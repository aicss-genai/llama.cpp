#!/usr/bin/env python3
"""Compare two or three benchmark CSV results (e.g. baseline vs optimized vs vulkan) and plot."""
import argparse
import csv
import sys
from pathlib import Path

def parse_csv(path):
    """Parse a benchmark CSV, skipping the two header rows (which may span multiple lines due to quoted newlines)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        all_rows = list(reader)
    # Find the first data row: starts with a model name (alpha char), skip header rows
    data_start = 0
    for i, r in enumerate(all_rows):
        if r and r[0].strip() and r[0].strip()[0].isalpha() and r[4].strip() in ("pp", "tg"):
            data_start = i
            break
    for r in all_rows[data_start:]:
        if not r or not r[0].strip():
            continue
        task = r[4].strip().replace("\n", "").replace("\r", "")
        if task not in ("pp", "tg"):
            continue
        rows.append({
            "model": r[0].strip(),
            "params": r[1].strip(),
            "quant": r[2].strip(),
            "arch": r[3].strip(),
            "task": task,
            "tokens": int(r[5].strip()),
            "toks": float(r[6].strip()),
            "stddev": float(r[7].strip()) if r[7].strip() else 0.0,
        })
    return rows

def main():
    ap = argparse.ArgumentParser(description="Compare two or three benchmark CSVs")
    ap.add_argument("baseline", help="Path to baseline CSV")
    ap.add_argument("optimized", help="Path to optimized CSV")
    ap.add_argument("--third", help="Path to optional third CSV (e.g. Vulkan)")
    ap.add_argument("-o", "--output", help="Output image path (svg/png)")
    ap.add_argument("--baseline-label", default="Baseline SYCL")
    ap.add_argument("--optimized-label", default="Optimized SYCL")
    ap.add_argument("--third-label", default="Vulkan")
    args = ap.parse_args()

    base = parse_csv(args.baseline)
    opt = parse_csv(args.optimized)
    third = parse_csv(args.third) if args.third else []

    # Build lookups: (model, task, tokens) -> row
    opt_map = {(r["model"], r["task"], r["tokens"]): r for r in opt}
    third_map = {(r["model"], r["task"], r["tokens"]): r for r in third}

    # Collect all unique keys across all datasets
    all_keys = set()
    for r in base:
        all_keys.add((r["model"], r["task"], r["tokens"]))
    for r in opt:
        all_keys.add((r["model"], r["task"], r["tokens"]))
    for r in third:
        all_keys.add((r["model"], r["task"], r["tokens"]))
    base_map = {(r["model"], r["task"], r["tokens"]): r for r in base}

    # Print comparison table
    if third:
        print(f"{'Model':<22} {'Task':>4} {'Tokens':>6}  "
              f"{args.baseline_label+' (tok/s)':>22}  "
              f"{args.optimized_label+' (tok/s)':>22}  "
              f"{args.third_label+' (tok/s)':>22}")
        print("-" * 110)
    else:
        print(f"{'Model':<22} {'Task':>4} {'Tokens':>6}  "
              f"{args.baseline_label+' (tok/s)':>22}  "
              f"{args.optimized_label+' (tok/s)':>22}  {'Speedup':>8}")
        print("-" * 96)

    comparisons = []
    # Sort keys for consistent output
    sorted_keys = sorted(all_keys, key=lambda k: (k[0], 0 if k[1] == "pp" else 1, k[2]))
    for key in sorted_keys:
        b = base_map.get(key)
        o = opt_map.get(key)
        t = third_map.get(key)
        # Need at least two of three to compare
        if sum(x is not None for x in [b, o, t]) < 2:
            continue

        model, task, tokens = key
        b_toks = b["toks"] if b else 0
        b_std = b["stddev"] if b else 0
        o_toks = o["toks"] if o else 0
        o_std = o["stddev"] if o else 0
        t_toks = t["toks"] if t else 0
        t_std = t["stddev"] if t else 0

        speedup = o_toks / b_toks if b_toks > 0 and o_toks > 0 else 0

        if third:
            b_str = f"{b_toks:>18.2f} ±{b_std:<5.2f}" if b else f"{'—':>24}"
            o_str = f"{o_toks:>18.2f} ±{o_std:<5.2f}" if o else f"{'—':>24}"
            t_str = f"{t_toks:>18.2f} ±{t_std:<5.2f}" if t else f"{'—':>24}"
            print(f"{model:<22} {task:>4} {tokens:>6}  {b_str}{o_str}{t_str}")
        else:
            print(f"{model:<22} {task:>4} {tokens:>6}  "
                  f"{b_toks:>18.2f} ±{b_std:<5.2f}"
                  f"{o_toks:>18.2f} ±{o_std:<5.2f}"
                  f"{speedup:>7.2f}x")

        entry = {
            "model": model, "task": task, "tokens": tokens,
            "toks": b_toks, "stddev": b_std,
            "opt_toks": o_toks, "opt_stddev": o_std,
            "third_toks": t_toks, "third_stddev": t_std,
            "speedup": speedup,
        }
        comparisons.append(entry)

    # Summary
    pp_rows = [c for c in comparisons if c["task"] == "pp"]
    tg_rows = [c for c in comparisons if c["task"] == "tg"]
    if pp_rows:
        sp = [c["speedup"] for c in pp_rows if c["speedup"] > 0]
        if sp:
            print(f"\nAverage prefill speedup ({args.optimized_label}/{args.baseline_label}):  {sum(sp)/len(sp):.2f}x")
    if tg_rows:
        sp = [c["speedup"] for c in tg_rows if c["speedup"] > 0]
        if sp:
            print(f"Average decode  speedup ({args.optimized_label}/{args.baseline_label}):  {sum(sp)/len(sp):.2f}x")

    if not args.output:
        return

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plot", file=sys.stderr)
        return

    models = sorted(set(c["model"] for c in comparisons))
    tasks = ["pp", "tg"]
    has_third = any(c["third_toks"] > 0 for c in comparisons)
    n_backends = 3 if has_third else 2

    fig, axes = plt.subplots(len(models), len(tasks), figsize=(14, 5 * len(models)),
                              squeeze=False, sharey=False)
    if has_third:
        fig.suptitle(f"{args.baseline_label} vs {args.optimized_label} vs {args.third_label}",
                     fontsize=16, y=0.98)
    else:
        fig.suptitle(f"{args.baseline_label} vs {args.optimized_label}", fontsize=16, y=0.98)

    bar_width = 0.8 / n_backends
    colors = {"baseline": "#5B9BD5", "optimized": "#ED7D31", "third": "#70AD47"}

    for mi, model in enumerate(models):
        for ti, task in enumerate(tasks):
            ax = axes[mi][ti]
            rows = sorted([c for c in comparisons if c["model"] == model and c["task"] == task],
                          key=lambda c: c["tokens"])
            if not rows:
                ax.set_visible(False)
                continue

            x = np.arange(len(rows))
            labels = [str(r["tokens"]) for r in rows]
            base_vals = [r["toks"] for r in rows]
            base_err = [r["stddev"] for r in rows]
            opt_vals = [r["opt_toks"] for r in rows]
            opt_err = [r["opt_stddev"] for r in rows]

            if has_third:
                third_vals = [r["third_toks"] for r in rows]
                third_err = [r["third_stddev"] for r in rows]
                offsets = np.array([-bar_width, 0, bar_width])
            else:
                offsets = np.array([-bar_width / 2, bar_width / 2])

            # Only draw bars for backends with data
            bi = 0
            if any(v > 0 for v in base_vals):
                ax.bar(x + offsets[0], base_vals, bar_width, yerr=base_err,
                       label=args.baseline_label, color=colors["baseline"], capsize=3)
            if any(v > 0 for v in opt_vals):
                ax.bar(x + offsets[1], opt_vals, bar_width, yerr=opt_err,
                       label=args.optimized_label, color=colors["optimized"], capsize=3)
            if has_third and any(v > 0 for v in third_vals):
                ax.bar(x + offsets[2], third_vals, bar_width, yerr=third_err,
                       label=args.third_label, color=colors["third"], capsize=3)

            # Annotate peak value per group
            for i, r in enumerate(rows):
                vals = [v for v in [r["toks"], r["opt_toks"], r["third_toks"]] if v > 0]
                if vals:
                    y_max = max(vals)
                    ax.text(i, y_max * 1.05, f"{y_max:.0f}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

            task_label = "Prefill (pp)" if task == "pp" else "Decode (tg)"
            ax.set_title(f"{model} — {task_label}", fontsize=12)
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Throughput (tok/s)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(fontsize=9)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {args.output}")

if __name__ == "__main__":
    main()
