"""DINO-space Fréchet distance (the FVD stand-in, see worldoptbench/metrics/fvd.py) per configuration.

    python scripts/analyze_fvd.py results/wan_fvd
    python scripts/analyze_fvd.py results/wan_fvd --components 8 --out results/wan_fvd.md

Needs sweep JSONs written with `PerceptualScorer(keep_descriptor=True)` (sweep_wan.py does that): each result holds
the video's descriptor and the baseline video's. For every configuration the distance is computed between the set of
baseline videos and the set of that configuration's videos, in the PCA space of the baseline set (default 8
components, covariance shrinkage 0.1). With 32 videos per set the estimate is noisy and biased upward, so:

  * the perturbation controls (perturb_*) give the floor (what a numerically equivalent change does at this n), and
    the column that matters is the excess over it;
  * a bootstrap over clips (resampling the 32 pairs with replacement) gives a spread, shown as `sd`.

This is NOT standard FVD (different network, different descriptor); do not compare it with published numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldoptbench.metrics.fvd import reduced_frechet_distance


def load(directory: Path) -> dict[str, dict[str, dict]]:
    configs: dict[str, dict[str, dict]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json" or path.name.startswith("."):
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            continue
        perceptual = (rows[0].get("visual") or {}).get("perceptual") or {}
        if "descriptor" not in perceptual or "descriptor_ref" not in perceptual:
            continue
        configs[rows[0].get("optimization") or path.stem] = {r["prompt_id"]: r for r in rows}
    return configs


def distance_with_spread(reference: np.ndarray, other: np.ndarray, components: int, shrink: float, resamples: int, seed: int = 0):
    point = reduced_frechet_distance(reference, other, components, shrink)
    rng = np.random.default_rng(seed)
    n = len(reference)
    draws = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        if len(set(idx.tolist())) < max(components + 1, 4):
            continue
        draws.append(reduced_frechet_distance(reference[idx], other[idx], components, shrink))
    return point, (st.stdev(draws) if len(draws) > 1 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--components", type=int, default=8)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--resamples", type=int, default=100)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    configs = load(args.directory)
    if not configs:
        print("no results with descriptors found in", args.directory)
        return 1

    rows = []
    for name, clips in configs.items():
        ids = sorted(clips)
        reference = np.array([clips[i]["visual"]["perceptual"]["descriptor_ref"] for i in ids])
        other = np.array([clips[i]["visual"]["perceptual"]["descriptor"] for i in ids])
        fd, sd = distance_with_spread(reference, other, args.components, args.shrink, args.resamples)
        rows.append({
            "name": name, "n": len(ids), "fd": fd, "sd": sd, "control": name.startswith("perturb_"),
            "speed": st.mean(clips[i]["speed"]["speedup"] for i in ids),
        })
    floor = st.mean(r["fd"] for r in rows if r["control"]) if any(r["control"] for r in rows) else None
    rows.sort(key=lambda r: (not r["control"], r["speed"]))

    lines = [
        (f"Frechet distance to the baseline set in the baseline's PCA space ({args.components} components, shrink {args.shrink}); "
        f"floor (mean over the perturbation controls) {(f'{floor:.3f}') if floor is not None else 'n/a'}; sd = bootstrap over clips."),
        "",
        "| configuration | n | speedup | distance | sd | excess over floor |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        excess = "control" if r["control"] else ("" if floor is None else f"{r['fd'] - floor:+.3f}")
        lines.append(f"| {r['name']} | {r['n']} | {r['speed']:.2f} | {r['fd']:.3f} | {r['sd']:.3f} | {excess} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
