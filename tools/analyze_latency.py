"""Summarize CHAT_LATENCY logs from backend and agent; standard library only.

python tools/analyze_latency.py backend.log agent.log --top 25
python tools/analyze_latency.py backend.log agent.log --trace UUID
Durations are inclusive: NEVER add nested stages to calculate end-to-end latency.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def read_rows(paths):
    seen = set()
    for path in paths:
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            # Render JSON export or plain log lines.
            try:
                outer = json.loads(line)
                if isinstance(outer, dict) and isinstance(outer.get('message'), str):
                    line = outer['message']
            except ValueError:
                pass
            if 'CHAT_LATENCY ' not in line:
                continue
            try:
                row = json.loads(line.split('CHAT_LATENCY ', 1)[1])
                key = row['component'], row['trace_id'], row['span_id']
                if key not in seen and isinstance(row['duration_ms'], (int, float)):
                    seen.add(key)
                    yield row
            except (ValueError, KeyError, TypeError):
                continue


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path, nargs='+')
    parser.add_argument('--trace')
    parser.add_argument('--top', type=int, default=30)
    args = parser.parse_args()
    rows = list(read_rows(args.logs))
    if args.trace:
        rows = [r for r in rows if r['trace_id'] == args.trace]
        for row in sorted(rows, key=lambda r: r['started_at']):
            print(f"{row['started_at']} {row['component']:7} {row['stage']:55} "
                  f"{row['duration_ms']:10.2f} ms {row['outcome']:10} "
                  f"{row.get('purpose', row.get('tool', ''))} "
                  f"span={row['span_id']} parent={row.get('parent_span_id', '-')}")
    else:
        groups = defaultdict(list)
        for row in rows:
            label = row['component'], row['stage'], row.get('purpose', row.get('tool', ''))
            groups[label].append(row)
        print('Inclusive stage durations; overlapping spans and cumulative intervals must not be added.\n')
        print('| Component / stage / purpose | Calls | Traces | p50 ms | p95 ms | Max ms | Non-ok |')
        print('|---|---:|---:|---:|---:|---:|---:|')
        for label, items in sorted(groups.items(), key=lambda g: max(x['duration_ms'] for x in g[1]), reverse=True)[:args.top]:
            values = [i['duration_ms'] for i in items]
            print(f"| {' / '.join(filter(None, label))} | {len(items)} | {len({i['trace_id'] for i in items})} | "
                  f"{percentile(values, .5):.2f} | {percentile(values, .95):.2f} | {max(values):.2f} | "
                  f"{sum(i['outcome'] != 'ok' for i in items)} |")
    print(f'\n{len(rows)} measurements, {len({r["trace_id"] for r in rows})} traces.')


if __name__ == '__main__':
    main()
