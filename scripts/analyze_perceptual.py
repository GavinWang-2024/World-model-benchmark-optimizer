"""Learned perceptual scores (DINO content similarity, DINO temporal consistency, CLIP prompt match) per
configuration, calibrated against the numerical-perturbation control.

    python scripts/analyze_perceptual.py results/wan_perceptual
    python scripts/analyze_perceptual.py results/wan_perceptual --out results/wan_perceptual.md

Reads the per-configuration JSON a sweep_wan.py run wrote (each result's `visual.perceptual`, see
worldoptbench/metrics/perceptual.py). Columns:

  * dino sim: mean +- sem over clips of the DINOv2 similarity between the video and the baseline video at
    matching frames (1.0 = identical content and layout); median and the worst clip alongside, since the mean
    hides a configuration that is fine on most clips and collapses on a few;
  * below floor: clips whose similarity is under the 5th percentile of the perturbation controls' (pooled
    over perturb_*), i.e. further from the baseline than a numerically equivalent change ever put a clip;
  * vs noise: clip-paired mean difference to the larger-effect control: `same` (within two standard errors),
    `LOWER` (less similar to the baseline than the control) or `higher`;
  * consistency delta: the video's own DINO temporal consistency minus the baseline video's (negative =
    flicker / jitter / collapse the baseline does not have);
  * clip delta: CLIP prompt score minus the baseline video's score (negative = matches the prompt less well).

DINO and CLIP are general image encoders: they see content and layout, not physics, and CLIP is weak on counts
and fine detail. A better screen than pixel statistics, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = st.mean(values)
    return mean, (st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0)


def percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, max(0, int(q * (len(sorted_values) - 1))))]


def load(directory: Path) -> dict[str, dict[str, dict]]:
    """config name -> clip id -> result row, for sweep JSONs that carry perceptual scores."""
    configs: dict[str, dict[str, dict]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json" or path.name.startswith("."):
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows or not (rows[0].get("visual") or {}).get("perceptual"):
            continue
        configs[rows[0].get("optimization") or path.stem] = {r["prompt_id"]: r for r in rows}
    return configs


def field(row: dict, key: str) -> float | None:
    return (row["visual"].get("perceptual") or {}).get(key)


def build(configs: dict[str, dict[str, dict]]) -> tuple[list[dict], float | None, str | None]:
    controls = {n: c for n, c in configs.items() if n.startswith("perturb_")}
    pooled = sorted(v for c in controls.values() for r in c.values() if (v := field(r, "dino_similarity")) is not None)
    floor = percentile(pooled, 0.05) if pooled else None
    null_name = min(controls, key=lambda n: st.mean(field(r, "dino_similarity") for r in controls[n].values())) if controls else None
    null = configs[null_name] if null_name else None

    rows = []
    for name, clips in configs.items():
        sims = [v for r in clips.values() if (v := field(r, "dino_similarity")) is not None]
        if not sims:
            continue
        sim_mean, sim_sem = mean_sem(sims)
        ordered = sorted(sims)
        verdict = ""
        if null is not None and name != null_name and not name.startswith("perturb_"):
            shared = sorted(set(clips) & set(null))
            diffs = [field(clips[c], "dino_similarity") - field(null[c], "dino_similarity") for c in shared]
            d_mean, d_sem = mean_sem(diffs)
            verdict = "LOWER" if d_mean < -2 * d_sem else ("higher" if d_mean > 2 * d_sem else "same")
        consistency = [
            field(r, "dino_consistency") - field(r, "dino_consistency_ref")
            for r in clips.values() if field(r, "dino_consistency") is not None and field(r, "dino_consistency_ref") is not None
        ]
        clip_delta = [v for r in clips.values() if (v := field(r, "clip_delta")) is not None]
        rows.append({
            "name": name, "n": len(sims), "control": name.startswith("perturb_"),
            "speed": st.mean(r["speed"]["speedup"] for r in clips.values()),
            "sim": sim_mean, "sim_sem": sim_sem, "median": st.median(sims), "worst": ordered[0],
            "below": sum(v < floor for v in sims) if floor is not None else None, "verdict": verdict,
            "consistency": st.mean(consistency) if consistency else None,
            "clip": mean_sem(clip_delta) if clip_delta else None,
        })
    rows.sort(key=lambda r: (not r["control"], r["speed"]))
    return rows, floor, null_name


def render(rows: list[dict], floor: float | None, null_name: str | None) -> str:
    lines = [
        f"Noise control: {null_name or 'none'}; floor (5th percentile of the controls' DINO similarity): "
        + (f"{floor:.3f}" if floor is not None else "n/a") + "; mean +- sem over clips.",
        "",
        "| configuration | n | speedup | dino sim | median | worst | below floor | vs noise | consistency delta | clip delta |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        clip = "" if r["clip"] is None else f"{r['clip'][0]:+.3f} +- {r['clip'][1]:.3f}"
        consistency = "" if r["consistency"] is None else f"{r['consistency']:+.3f}"
        below = "" if r["below"] is None else f"{r['below']}/{r['n']}"
        lines.append(
            f"| {r['name']} | {r['n']} | {r['speed']:.2f} | {r['sim']:.3f} +- {r['sim_sem']:.3f} | {r['median']:.3f} | "
            f"{r['worst']:.3f} | {below} | {r['verdict'] or ('control' if r['control'] else '')} | {consistency} | {clip} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directories", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    configs: dict[str, dict[str, dict]] = {}
    for directory in args.directories:
        configs.update(load(directory))
    if not configs:
        print("no results with perceptual scores found in", ", ".join(map(str, args.directories)))
        return 1
    rows, floor, null_name = build(configs)
    table = render(rows, floor, null_name)
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
