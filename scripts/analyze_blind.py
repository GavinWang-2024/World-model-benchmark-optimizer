"""Summarize a sweep's reference-free statistics against the numerical-perturbation control.

    python scripts/analyze_blind.py results/wan_blind

Reads every <config>.json the sweep wrote (they carry per-clip `visual.blind_ratio` and
`visual.style_deviation`, see worldoptbench/metrics/blind.py) and prints, per configuration:

  * speedup and PSNR (for orientation),
  * style deviation: mean |log ratio| of seven pixel statistics against the baseline video
    (0 = same statistics), mean +- sem over clips,
  * "vs noise": the clip-paired difference between this configuration's deviation and the
    perturbation control's (perturb_*; the larger of those is the null). Positive and larger than
    twice its sem means the configuration moves the video's statistics more than a numerically
    equivalent change does, i.e. a shift beyond the metric's noise floor,
  * the mean log ratio of each statistic (bias: which way it moved; 0.00 = no shift, -0.69 = halved).

Without a perturb_* control only the first three columns are meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldoptbench.metrics.blind import STATS  # noqa: E402


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = st.mean(values)
    return mean, (st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0)


def load(directory: Path) -> dict[str, dict[str, dict]]:
    """config name -> clip id -> result row, for every sweep JSON in the directory."""
    configs: dict[str, dict[str, dict]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in ("summary.json",) or path.name.startswith("."):
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows or "visual" not in rows[0]:
            continue
        if rows[0]["visual"].get("style_deviation") is None:
            continue
        configs[rows[0].get("optimization") or path.stem] = {r["prompt_id"]: r for r in rows}
    return configs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    configs = load(args.directory)
    if not configs:
        print("no results with reference-free statistics in", args.directory)
        return 1

    controls = {n: c for n, c in configs.items() if n.startswith("perturb_")}
    null_name = max(controls, key=lambda n: st.mean(r["visual"]["style_deviation"] for r in controls[n].values())) if controls else None
    null = configs[null_name] if null_name else None

    short = {"sharpness": "sharp", "noise": "noise", "contrast": "contr", "colorfulness": "color", "motion": "motion", "flicker": "flick", "jerk": "jerk"}
    print(f"null control: {null_name or 'none'}")
    header = f"{'configuration':20s} {'speedup':>8s} {'PSNR':>6s} {'style dev':>14s} {'median':>7s} {'p90':>6s} {'>0.5':>5s} {'vs noise (paired)':>19s}  " + " ".join(f"{short[k]:>6s}" for k in STATS)
    print(header)
    order = sorted(configs, key=lambda n: (n.startswith("perturb_") is False, st.mean(r["speed"]["speedup"] for r in configs[n].values())))
    for name in order:
        rows = configs[name]
        speed = st.mean(r["speed"]["speedup"] for r in rows.values())
        psnr = st.mean(min(r["visual"]["psnr"], 60.0) for r in rows.values() if r["visual"].get("psnr") is not None)
        dev_mean, dev_sem = mean_sem([r["visual"]["style_deviation"] for r in rows.values()])
        versus = ""
        if null is not None and name != null_name:
            shared = sorted(set(rows) & set(null))
            diffs = [rows[c]["visual"]["style_deviation"] - null[c]["visual"]["style_deviation"] for c in shared]
            d_mean, d_sem = mean_sem(diffs)
            flag = "BEYOND" if d_mean > 2 * d_sem else ("below" if d_mean < -2 * d_sem else "within")
            versus = f"{d_mean:+.3f}+-{d_sem:.3f} {flag}"
        bias = [mean_sem([math.log(r["visual"]["blind_ratio"][k]) for r in rows.values()])[0] for k in STATS]
        devs = sorted(r["visual"]["style_deviation"] for r in rows.values())
        median, p90, broken = st.median(devs), devs[int(0.9 * (len(devs) - 1))], sum(d > 0.5 for d in devs)
        print(
            f"{name:20s} {speed:8.2f} {psnr:6.1f} {dev_mean:7.3f}+-{dev_sem:.3f} {median:7.3f} {p90:6.3f} {broken:5d} {versus:>19s}  "
            + " ".join(f"{b:+6.2f}" for b in bias)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
