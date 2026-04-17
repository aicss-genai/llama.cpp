#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_events(path: Path):
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported trace format in {path}")


def get_x_event_bounds(events):
    x_events = [ev for ev in events if ev.get("ph") == "X" and "dur" in ev and "ts" in ev]
    if not x_events:
        raise ValueError("trace contains no complete X events with timestamps")

    start_us = min(float(ev["ts"]) for ev in x_events)
    end_us = max(float(ev["ts"]) + float(ev["dur"]) for ev in x_events)
    return start_us, end_us


def resolve_window(events, start_ms=None, end_ms=None, start_pct=None, end_pct=None):
    trace_start_us, trace_end_us = get_x_event_bounds(events)
    span_us = trace_end_us - trace_start_us

    if start_ms is not None and start_pct is not None:
        raise ValueError("use either start_ms or start_pct, not both")
    if end_ms is not None and end_pct is not None:
        raise ValueError("use either end_ms or end_pct, not both")

    window_start_us = trace_start_us
    window_end_us = trace_end_us

    if start_ms is not None:
        window_start_us = trace_start_us + start_ms * 1000.0
    elif start_pct is not None:
        window_start_us = trace_start_us + span_us * (start_pct / 100.0)

    if end_ms is not None:
        window_end_us = trace_start_us + end_ms * 1000.0
    elif end_pct is not None:
        window_end_us = trace_start_us + span_us * (end_pct / 100.0)

    if window_start_us >= window_end_us:
        raise ValueError("window start must be earlier than window end")

    return window_start_us, window_end_us, trace_start_us, trace_end_us


def summarize(events, category=None, limit=25, start_ms=None, end_ms=None, start_pct=None, end_pct=None):
    totals = defaultdict(lambda: [0.0, 0])
    window_start_us, window_end_us, trace_start_us, trace_end_us = resolve_window(
        events,
        start_ms=start_ms,
        end_ms=end_ms,
        start_pct=start_pct,
        end_pct=end_pct,
    )

    for ev in events:
        if ev.get("ph") != "X" or "dur" not in ev or "ts" not in ev:
            continue
        cat = ev.get("cat", "")
        if category is not None and cat != category:
            continue

        ev_start_us = float(ev["ts"])
        ev_end_us = ev_start_us + float(ev["dur"])
        overlap_us = min(ev_end_us, window_end_us) - max(ev_start_us, window_start_us)
        if overlap_us <= 0:
            continue

        name = ev.get("name", "")
        totals[(cat, name)][0] += overlap_us
        totals[(cat, name)][1] += 1

    items = sorted(totals.items(), key=lambda kv: kv[1][0], reverse=True)[:limit]
    return {
        "trace_start_us": trace_start_us,
        "trace_end_us": trace_end_us,
        "window_start_us": window_start_us,
        "window_end_us": window_end_us,
        "rows": [
        {
            "category": cat,
            "name": name,
            "total_ms": dur_us / 1000.0,
            "calls": calls,
            "avg_ms": (dur_us / calls) / 1000.0,
        }
        for (cat, name), (dur_us, calls) in items
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize chrome-format unitrace events, optionally clipped to a time window."
    )
    parser.add_argument("trace_json", help="path to the unitrace chrome JSON")
    parser.add_argument("category", nargs="?", default="all", help="all, gpu_op, or cpu_op")
    parser.add_argument("limit", nargs="?", type=int, default=25, help="maximum rows to print")
    parser.add_argument("--start-ms", type=float, default=None, help="window start relative to first X event, in ms")
    parser.add_argument("--end-ms", type=float, default=None, help="window end relative to first X event, in ms")
    parser.add_argument("--start-pct", type=float, default=None, help="window start as percent of X-event trace span")
    parser.add_argument("--end-pct", type=float, default=None, help="window end as percent of X-event trace span")
    args = parser.parse_args()

    path = Path(args.trace_json)
    category = None if args.category == "all" else args.category

    summary = summarize(
        load_events(path),
        category=category,
        limit=args.limit,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        start_pct=args.start_pct,
        end_pct=args.end_pct,
    )

    trace_start_ms = summary["trace_start_us"] / 1000.0
    trace_end_ms = summary["trace_end_us"] / 1000.0
    window_start_ms = summary["window_start_us"] / 1000.0
    window_end_ms = summary["window_end_us"] / 1000.0
    print(
        f"# window_ms={window_start_ms - trace_start_ms:.3f}..{window_end_ms - trace_start_ms:.3f} "
        f"within total_ms={trace_end_ms - trace_start_ms:.3f}"
    )
    for row in summary["rows"]:
        print(
            f"{row['total_ms']:10.3f} ms total | "
            f"{row['calls']:6d} calls | "
            f"{row['avg_ms']:9.3f} ms avg | "
            f"{row['category']}::{row['name']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
