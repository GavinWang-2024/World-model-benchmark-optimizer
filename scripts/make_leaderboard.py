"""Generate LEADERBOARD.md from the sweep results in this repo (outline section 12-F).

    python scripts/make_leaderboard.py                 # writes LEADERBOARD.md
    python scripts/make_leaderboard.py --out other.md

A static leaderboard: one table per model, every row a measured configuration, built from the result files rather
than typed by hand, so it cannot drift from the data. It is a snapshot of what was measured on one machine, not a
service: submitting a result means running the sweep scripts and sending the JSON. Sources:

  Dreamer  results/sweep_long_policy/summary.json   (scripts/sweep_dreamer.py; 515k-step walker, held-out walking)
  Wan2.1   results/wan_perceptual/                  (scripts/sweep_wan.py; DINO/CLIP scores, 32 clips)
           results/wan_all/                          (the pixel-statistics sweeps, for the style-deviation column)

A source that is missing is skipped with a note, so the script works on a fresh checkout that has only some results.
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def hardware() -> str:
    try:
        from worldoptbench.reporting import detect_hardware

        return detect_hardware()
    except Exception:  # noqa: BLE001
        return "unknown"


def dreamer_section(path: Path) -> list[str]:
    if not path.exists():
        return [f"_No Dreamer sweep found at `{path.relative_to(ROOT)}`; run `scripts/sweep_dreamer.py`._", ""]
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(summary.items(), key=lambda kv: kv[1]["speedup"])
    first = rows[0][1]
    lines = [
        f"{first['n_scenarios']} held-out walking scenarios at 2 / 5 / 10 / 20 s (`scripts/sweep_dreamer.py`, medians of 3 runs). "
        "Skill is simulator fidelity against freezing the last context frame (0 = no better than a frozen scene). "
        f"The eager baseline's latency varies by about +-{first['noise_pct']:.0f}% between runs, so speedups against it are indicative; "
        "compare the absolute latencies. Rows that use the TensorRT or Gumbel sampler (`trt`, `tensorrt`, `latent_noise`) draw different random "
        "numbers than the baseline, so their skill change is dominated by sampling noise at this many scenarios (about +-0.05): "
        "a change of that size is not evidence of lost fidelity (the sampler was checked separately and is statistically equivalent). "
        "Rows that keep the noise stream (CUDA graphs with channels_last, cuDNN benchmark, shared pool) are compared tightly, and "
        "low-rank factorization and sparse decoding visibly hurt. Drift of the baseline: "
        + (f"{first['mean_drift_rate']:+.3f} +- {first['drift_rate_sem']:.3f} skill per second." if first.get("mean_drift_rate") is not None else "n/a."),
        "",
        "| configuration | speedup | latency (ms) | skill | skill change | PAES | GPU held (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in rows:
        lines.append(
            f"| {name} | {r['speedup']:.2f} | {r['latency_ms']:.1f} | {r['skill']:.3f} | {r['d_skill']:+.3f} | "
            f"{r['paes']:.3f} | {r.get('reserved_mb', float('nan')):.0f} |"
        )
    return [*lines, ""]


def policy_section(coarse_dir: Path, fine_dir: Path) -> list[str]:
    """Policy-ranking agreement per optimized Dreamer world model (scripts/policy_ranking_dreamer.py), recomputed from the
    saved returns so the table cannot drift from them."""
    import statistics as st_

    from worldoptbench.utility import rank_agreement

    sets = {}
    for label, directory in (("coarse", coarse_dir), ("fine", fine_dir)):
        true_path, imagined_path = directory / "true_returns.json", directory / "imagined_returns.json"
        if true_path.exists() and imagined_path.exists():
            true = {k: st_.mean(v) for k, v in json.loads(true_path.read_text(encoding="utf-8"))["returns"].items()}
            sets[label] = (true, json.loads(imagined_path.read_text(encoding="utf-8")))
    if not sets:
        return ["_No policy-ranking results found; run `scripts/policy_ranking_dreamer.py`._", ""]
    names = list(next(iter(sets.values()))[1])
    lines = [
        "Does the optimized world model still rank policies like the real simulator and like the unoptimized model? Nine policies per set "
        "(variants of the trained actor: `coarse` spans terrible to good play, `fine` is graded action noise so true returns are close), "
        "imagined from 24 held-out walking start states over 60 steps (`scripts/policy_ranking_dreamer.py`). Spearman rank correlation; "
        "`baseline_reseed` is the unoptimized model with different random draws, the noise floor: nothing can agree with another model "
        "better than that. **With nine policies these measures cannot see small losses**: one swapped pair of policies moves Spearman by about "
        "0.02-0.03, so rows within a few hundredths of the reseed's value are not distinguishable from it.",
        "",
        "| model | coarse: vs simulator | coarse: vs unoptimized | fine: vs simulator | fine: vs unoptimized | actor's imagined return (coarse) |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        cells = []
        for label in ("coarse", "fine"):
            if label not in sets or name not in sets[label][1]:
                cells += ["", ""]
                continue
            true, imagined = sets[label]
            vs_true = rank_agreement(true, imagined[name]).spearman
            vs_base = None if name == "baseline" else rank_agreement(imagined["baseline"], imagined[name]).spearman
            cells += ["" if vs_true is None else f"{vs_true:+.2f}", "" if vs_base is None else f"{vs_base:+.2f}"]
        scale = ""
        if "coarse" in sets and name in sets["coarse"][1] and "actor" in sets["coarse"][1][name]:
            scale = f"{sets['coarse'][1][name]['actor']:.0f}"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {scale} |")
    lines += [
        "",
        "Read the last column with the others: `low_rank_0.25` keeps the ranking (about 0.95) while its imagined returns collapse "
        "(the actor scores about 40 against 112 unoptimized), so it is still usable to *order* policies and useless for estimating how "
        "good one is. The same model lost most of its physics skill (0.11 against 0.30) in the Dreamer table above.",
        "",
    ]
    return lines


def wan_section(perceptual_dirs: list[Path], pixel_dir: Path) -> list[str]:
    perceptual_dirs = [d for d in perceptual_dirs if d.exists()]
    if not perceptual_dirs:
        return ["_No Wan sweep found under `results/wan_perceptual`; run `scripts/sweep_wan.py`._", ""]
    from analyze_blind import load as load_pixel
    from analyze_perceptual import build, load

    configs: dict = {}
    for directory in perceptual_dirs:
        configs.update(load(directory))
    rows, floor, null_name = build(configs)
    pixel = load_pixel(pixel_dir) if pixel_dir.exists() else {}

    def style(name: str) -> str:
        clips = pixel.get(name) or configs.get(name)
        values = [r["visual"].get("style_deviation") for r in (clips or {}).values() if r["visual"].get("style_deviation") is not None]
        return f"{st.mean(values):.3f}" if values else ""

    lines = [
        "Wan2.1-T2V-1.3B, 32 clips (16 prompts x 2 seeds), 2 s at 192x320, 30 steps. **There is no physics score for text-to-video here**: "
        "quality is agreement with the unoptimized video for the same prompt and seed, measured with DINOv2 and CLIP "
        "(`worldoptbench/metrics/perceptual.py`). DINO similarity is sameness (1 = identical), not quality: read it with the "
        f"noise controls (a numerically equivalent change scores about 0.88; the floor, the controls' 5th percentile, is {floor:.2f}) "
        "and with the CLIP and consistency columns. Pixel style deviation is a coarse screen (reliable only below about 0.1).",
        "",
        "| configuration | speedup | DINO similarity | clips below floor | vs noise | CLIP delta | consistency delta | style deviation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        clip = "" if r["clip"] is None else f"{r['clip'][0]:+.3f}"
        consistency = "" if r["consistency"] is None else f"{r['consistency']:+.3f}"
        below = "" if r["below"] is None else f"{r['below']}/{r['n']}"
        lines.append(
            f"| {r['name']} | {r['speed']:.2f} | {r['sim']:.3f} | {below} | {r['verdict'] or ('control' if r['control'] else '')} | "
            f"{clip} | {consistency} | {style(r['name'])} |"
        )
    return [*lines, ""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "LEADERBOARD.md")
    ap.add_argument("--dreamer", type=Path, default=ROOT / "results" / "sweep_long_policy" / "summary.json")
    ap.add_argument("--wan", type=Path, nargs="+", default=[ROOT / "results" / d for d in ("wan_perceptual", "wan_more", "wan_more2", "wan_more3")])
    ap.add_argument("--wan-pixel", type=Path, default=ROOT / "results" / "wan_all")
    args = ap.parse_args()

    lines = [
        "# WorldOptBench leaderboard",
        "",
        f"Generated {datetime.date.today().isoformat()} by `scripts/make_leaderboard.py` from the result files in this repo. "
        f"Everything was measured on one machine ({hardware()}), so speedups say how configurations compare *here*; "
        "they are not predictions for other hardware. Two small models, not a general result across world models. "
        "Every row is a configuration that was actually run; `OPTIMIZATION_CATALOG.md` lists what was not.",
        "",
        "## Dreamer (DreamerV3 walker, a real world model with simulator ground truth)",
        "",
        *dreamer_section(args.dreamer),
        "## Dreamer: does the optimized model still rank policies correctly? (the outline's functional-utility axis)",
        "",
        *policy_section(ROOT / "results" / "policy_ranking", ROOT / "results" / "policy_ranking_fine"),
        "## Wan2.1-1.3B (text-to-video; the diffusion optimizations)",
        "",
        *wan_section(args.wan, args.wan_pixel),
        "## Reproducing a row",
        "",
        "```bash",
        "python scripts/sweep_dreamer.py --checkpoint ../dreamerv3-torch/logdir/walker_long --episodes ../dreamerv3-torch/logdir/walker_long/eval_eps --min-return 600",
        "python scripts/sweep_wan.py --standard-set worldoptbench/prompts/wan_set_16.json --out-dir results/wan_perceptual",
        "python scripts/analyze_perceptual.py results/wan_perceptual",
        "```",
        "",
        "Details, caveats and the negative results are in `DREAMER_SETUP.md` and `DESIGN_DIFFUSION.md`.",
    ]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
