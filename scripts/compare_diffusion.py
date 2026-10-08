"""One comparison table across diffusion sweeps (the plan's "baseline / WorldCache / AdaCache / ..."
table, on speed and quality), with the noise control and the Pareto front marked.

    python scripts/compare_diffusion.py results/wan_blind results/wan_blind_ada results/wan_blind_warp
    python scripts/compare_diffusion.py results/wan_blind results/wan_blind_ada --out results/wan_comparison.md

Each directory holds the per-configuration JSON a sweep_wan.py run wrote. Later directories override
earlier ones for the same configuration name. Quality columns:

  * PSNR vs the unoptimized video (saturates at a noise floor, see DESIGN_DIFFUSION.md),
  * style deviation: mean |log ratio| of seven reference-free pixel statistics vs the baseline video
    (0 = same statistics),
  * "vs noise": clip-paired style deviation minus the numerical-perturbation control's: `within` /
    `below` mean indistinguishable from (or closer to the baseline than) a bf16-level perturbation,
    `BEYOND` a shift larger than that.

PAES is not used here: it needs a physics score, and for text-to-video there is none; "style
deviation" is the closest reference-free substitute and is not a physics measure.

`pareto` marks configurations no other row beats on both speedup (higher) and style deviation (lower).
Controls (perturb_*) are shown first and excluded from the front. The mean hides bimodal failures
(a configuration that is perfect on most clips and collapses on a few), so the median, the 90th percentile and
the number of clips above 0.5 are shown too. Contact sheets (scripts/make_contact_sheets.py) showed that
style deviation below ~0.1 is reliably baseline-like but above ~0.15 does not separate broken from merely
different; treat this table as a coarse screen.
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_blind import load, mean_sem


def pareto_front(points: dict[str, tuple[float, float]]) -> set[str]:
    """Names not dominated by another on (speedup higher, deviation lower)."""
    front = set()
    for name, (speed, dev) in points.items():
        dominated = any(
            other != name and s >= speed and d <= dev and (s > speed or d < dev)
            for other, (s, d) in points.items()
        )
        if not dominated:
            front.add(name)
    return front


def build_rows(configs: dict[str, dict[str, dict]]) -> tuple[list[dict], str | None]:
    controls = {n: c for n, c in configs.items() if n.startswith("perturb_")}
    null_name = max(controls, key=lambda n: st.mean(r["visual"]["style_deviation"] for r in controls[n].values())) if controls else None
    null = configs[null_name] if null_name else None

    rows = []
    for name, clips in configs.items():
        speed, speed_sem = mean_sem([r["speed"]["speedup"] for r in clips.values()])
        psnrs = [min(r["visual"]["psnr"], 60.0) for r in clips.values() if r["visual"].get("psnr") is not None]
        dev, dev_sem = mean_sem([r["visual"]["style_deviation"] for r in clips.values()])
        verdict = ""
        if null is not None and name != null_name and not name.startswith("perturb_"):
            shared = sorted(set(clips) & set(null))
            diffs = [clips[c]["visual"]["style_deviation"] - null[c]["visual"]["style_deviation"] for c in shared]
            d_mean, d_sem = mean_sem(diffs)
            verdict = "BEYOND" if d_mean > 2 * d_sem else ("below" if d_mean < -2 * d_sem else "within")
        devs = sorted(r["visual"]["style_deviation"] for r in clips.values())
        rows.append({
            "median": st.median(devs), "p90": devs[int(0.9 * (len(devs) - 1))], "broken": sum(d > 0.5 for d in devs),
            "name": name, "n": len(clips), "speed": speed, "speed_sem": speed_sem,
            "psnr": st.mean(psnrs) if psnrs else math.nan, "dev": dev, "dev_sem": dev_sem,
            "verdict": verdict, "control": name.startswith("perturb_"),
        })
    candidates = {r["name"]: (r["speed"], r["dev"]) for r in rows if not r["control"]}
    front = pareto_front(candidates)
    for r in rows:
        r["pareto"] = r["name"] in front
    rows.sort(key=lambda r: (not r["control"], r["speed"]))
    return rows, null_name


def render(rows: list[dict], null_name: str | None) -> str:
    lines = [
        f"Noise control: {null_name or 'none (no perturb_* results found)'}; n = clips per row; mean +- sem.",
        "",
        "| configuration | n | speedup | PSNR (dB) | style deviation | median | p90 | clips > 0.5 | vs noise | pareto |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['n']} | {r['speed']:.2f} +- {r['speed_sem']:.2f} | {r['psnr']:.1f} | "
            f"{r['dev']:.3f} +- {r['dev_sem']:.3f} | {r['median']:.3f} | {r['p90']:.3f} | {r['broken']} | "
            f"{r['verdict'] or ('control' if r['control'] else '')} | "
            f"{'yes' if r['pareto'] else ''} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directories", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="also write the table to this markdown file")
    args = ap.parse_args()

    configs: dict[str, dict[str, dict]] = {}
    for directory in args.directories:
        configs.update(load(directory))
    if not configs:
        print("no results with reference-free statistics found in", ", ".join(map(str, args.directories)))
        return 1
    rows, null_name = build_rows(configs)
    table = render(rows, null_name)
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
