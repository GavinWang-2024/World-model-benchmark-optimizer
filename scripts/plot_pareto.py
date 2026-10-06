"""Pareto plots of the diffusion sweeps: speedup against how much of the baseline video survives.

    python scripts/plot_pareto.py results/wan_perceptual --out results/wan_pareto.png

Left panel: speedup (x) against DINOv2 similarity to the baseline video (y; 1 = identical content), with the
numerical-perturbation controls drawn as a band (mean +- their spread) and the 5th-percentile floor as a dashed
line, and the Pareto front (no configuration beats it on both axes) as a step line. Right panel: speedup against the
share of clips that fall below that floor, i.e. further from the baseline than a numerically equivalent change ever put
a clip. Both come from `analyze_perceptual.py`'s data (sweep JSONs with `visual.perceptual`).

Families are colour-coded by name. DINO similarity measures sameness, not quality; see DESIGN_DIFFUSION.md.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_perceptual import build, load  # noqa: E402

FAMILIES = [
    ("cfg", "CFG truncation / PAB", "tab:green"),
    ("pab", "CFG truncation / PAB", "tab:green"),
    ("cfg0.6+wc", "WorldCache (+CFG truncation)", "tab:blue"),
    ("wc_", "WorldCache (+CFG truncation)", "tab:blue"),
    ("fbc", "first-block cache", "tab:orange"),
    ("cfg0.6+ada", "AdaCache with guard", "tab:purple"),
    ("ada_fast30_guard", "AdaCache with guard", "tab:purple"),
    ("ada", "AdaCache", "tab:red"),
    ("taylorseer", "TaylorSeer / fewer steps", "tab:brown"),
    ("steps", "TaylorSeer / fewer steps", "tab:brown"),
]


def family(name: str) -> tuple[str, str]:
    if name.startswith("perturb_"):
        return "noise control", "gray"
    if name.startswith("kv+uncond") or name.startswith("uncond"):
        return "unconditional-pass reuse", "tab:olive"
    if name.startswith("mag_"):
        return "MagCache (+ CFG truncation, KV cache, bf16 VAE)", "tab:cyan"
    if name.startswith("sched_") or name.startswith("s20shift"):
        return "scheduler + fewer steps", "tab:pink"
    if name.startswith(("lossless_", "vae_", "kv_cache", "tf32", "cudnn_bench")):
        return "lossless extras (KV cache, bf16 VAE, TF32)", "lightgreen"
    if name.startswith("kv+vae"):
        return "CFG truncation / PAB", "tab:green"
    if name.startswith("cfg0.6+wc") or name.startswith("wc_"):
        return "WorldCache (+CFG truncation)", "tab:blue"
    if name.startswith("cfg0.6+ada") or "guard" in name:
        return "AdaCache with guard", "tab:purple"
    for prefix, label, colour in FAMILIES:
        if name.startswith(prefix):
            return label, colour
    return "other", "black"


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points no other point beats on both axes (higher is better on both), sorted by x."""
    front = [p for p in points if not any(q != p and q[0] >= p[0] and q[1] >= p[1] for q in points)]
    return sorted(front)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directories", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/wan_pareto.png"))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs: dict = {}
    for directory in args.directories:
        configs.update(load(directory))
    if not configs:
        print("no results with perceptual scores found")
        return 1
    rows, floor, _ = build(configs)
    candidates = [r for r in rows if not r["control"]]
    controls = [r for r in rows if r["control"]]

    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6.2))
    if controls:
        sims = [r["sim"] for r in controls]
        mean_c = st.mean(sims)
        spread = max(abs(s - mean_c) for s in sims) + max(r["sim_sem"] for r in controls)
        left.axhspan(mean_c - spread, mean_c + spread, color="gray", alpha=0.25, label="noise control (numerically equivalent change)")
    if floor is not None:
        left.axhline(floor, color="gray", linestyle="--", linewidth=1, label=f"5th percentile of controls ({floor:.2f})")

    seen = set()
    for r in candidates:
        label, colour = family(r["name"])
        left.scatter(r["speed"], r["sim"], color=colour, s=45, label=None if label in seen else label, zorder=3)
        left.annotate(r["name"], (r["speed"], r["sim"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        right.scatter(r["speed"], 100.0 * r["below"] / r["n"], color=colour, s=45, label=None if label in seen else label, zorder=3)
        right.annotate(r["name"], (r["speed"], 100.0 * r["below"] / r["n"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        seen.add(label)

    front = pareto_front([(r["speed"], r["sim"]) for r in candidates])
    left.step([p[0] for p in front], [p[1] for p in front], where="post", color="black", linewidth=1.2, alpha=0.7, label="Pareto front")

    left.set_xlabel("speedup (x)")
    left.set_ylabel("DINOv2 similarity to the baseline video (1 = identical)")
    left.set_title("Speed against how much of the baseline survives")
    left.legend(fontsize=8, loc="lower left")
    left.grid(alpha=0.25)
    right.set_xlabel("speedup (x)")
    right.set_ylabel("% of clips below the noise floor")
    right.set_title("Share of clips further from the baseline than noise ever put one")
    right.grid(alpha=0.25)
    fig.suptitle("Wan2.1-1.3B, 32 clips: 16 prompts x 2 seeds, 2 s at 192x320 (higher-left is better on the left, lower-right on the right)", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
