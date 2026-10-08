"""Dollars per generated second of video for each configuration, and the cheapest one above a quality floor.

    python scripts/cost_pareto.py results/wan_perceptual --usd-per-hour 1.75 --min-dino 0.744
    python scripts/cost_pareto.py results/wan_perceptual --usd-per-hour 0.12 --hardware "this laptop (assumed $0.12/h of electricity)"

Latencies were measured on one machine; `--usd-per-hour` must be that machine's cost (see worldoptbench/cost.py for
what this does and does not mean). Quality is the DINOv2 similarity to the baseline video from the perceptual
scores (higher = closer to the baseline); `--min-dino` is the floor for "the cheapest configuration that stays
this close". With no floor given, the table is just cost.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldoptbench.cost import cheapest_meeting, cost_pareto, cost_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--usd-per-hour", type=float, required=True)
    ap.add_argument("--hardware", default="the machine the latencies were measured on")
    ap.add_argument("--min-dino", type=float, help="quality floor: DINO similarity to the baseline video")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    configs, quality = {}, {}
    for path in sorted(args.directory.glob("*.json")):
        if path.name == "summary.json" or path.name.startswith("."):
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows or "speed" not in rows[0]:
            continue
        name = rows[0].get("optimization") or path.stem
        configs[name] = rows
        sims = [(r["visual"].get("perceptual") or {}).get("dino_similarity") for r in rows]
        sims = [s for s in sims if s is not None]
        if sims:
            quality[name] = st.mean(sims)
    if not configs:
        print("no result files found in", args.directory)
        return 1

    rows = cost_rows(configs, args.usd_per_hour, quality)
    front = {r.name for r in cost_pareto(rows)}
    baseline = next((r for r in rows if r.speedup is not None and abs(r.speedup - 1.0) < 0.02 and not r.name.startswith("perturb_")), None)
    lines = [
        f"Cost at ${args.usd_per_hour:.2f}/hour for {args.hardware}; $ per generated second of video (compute only).",
        "",
        "| configuration | $ per generated second | vs unoptimized | speedup | DINO similarity | on cost/quality front |",
        "|---|---|---|---|---|---|",
    ]
    reference = baseline.usd_per_second if baseline else None
    for r in rows:
        ratio = "" if reference is None else f"{r.usd_per_second / reference:.2f}x"
        quality_text = "" if r.quality is None else f"{r.quality:.3f}"
        speed_text = "" if r.speedup is None else f"{r.speedup:.2f}"
        lines.append(f"| {r.name} | ${r.usd_per_second:.6f} | {ratio} | {speed_text} | {quality_text} | {'yes' if r.name in front else ''} |")
    if args.min_dino is not None:
        best = cheapest_meeting([r for r in rows if not r.name.startswith("perturb_")], args.min_dino)
        lines.append("")
        lines.append(
            f"Cheapest configuration with DINO similarity >= {args.min_dino}: "
            + (f"{best.name} at ${best.usd_per_second:.6f} per generated second ({best.speedup:.2f}x)." if best else "none qualifies.")
        )
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
