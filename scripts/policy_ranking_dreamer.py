"""Does an optimized Dreamer world model still rank policies the way the original does? (outline section 12-D)

    python scripts/policy_ranking_dreamer.py --checkpoint ../dreamerv3-torch/logdir/walker_long \\
        --episodes ../dreamerv3-torch/logdir/walker_long/eval_eps

Policies are variants of the trained actor (worldoptbench/models/dreamer_policy.py): the actor itself, the actor with
action noise, halved, held for two steps, inverted, plus zero and random actions. Three rankings of them are compared:

  true        mean return in the real simulator (the baseline world model does the state estimation, as the agent does)
  baseline    imagined return inside the unoptimized world model (start states from held-out walking, --starts of them,
              --horizon imagined steps each)
  optimized   the same imagined return inside each optimized world model

`baseline_reseed` is the unoptimized model again with different random draws: how much the ranking moves from sampling noise
alone, which is the yardstick for every other row (an optimization is only visibly worse than this if its agreement is lower).
The agreement measures are in worldoptbench/utility.py. Keep the GPU otherwise idle (the simulator renders on it).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from worldoptbench.models.dreamer import DreamerRepo, DreamerWorldModel
from worldoptbench.models.dreamer_policy import (
    POLICY_SETS,
    imagined_returns,
    load_actor,
    load_starts,
    true_returns,
)
from worldoptbench.stack import OptimizationStack
from worldoptbench.utility import rank_agreement, ranking

# name -> (modules, per-module kwargs, imagination seed)
CONFIGS: dict[str, tuple[list[str], dict, int]] = {
    "baseline": ([], {}, 0),
    "baseline_reseed": ([], {}, 1),
    "gumbel": (["gumbel_sampling"], {}, 0),
    "tensorrt_fp16": (["tensorrt"], {"tensorrt": {"precision": "fp16"}}, 0),
    "latent_noise_0.5": (["latent_noise"], {"latent_noise": {"scale": 0.5}}, 0),
    "latent_noise_0": (["latent_noise"], {"latent_noise": {"scale": 0.0}}, 0),
    "bf16_autocast": (["precision"], {"precision": {"dtype": "bfloat16"}}, 0),
    "int8_weight_only": (["quantization"], {"quantization": {"scheme": "int8_weight_only"}}, 0),
    "low_rank_0.5": (["low_rank"], {"low_rank": {"rank_fraction": 0.5}}, 0),
    "low_rank_0.25": (["low_rank"], {"low_rank": {"rank_fraction": 0.25}}, 0),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--episodes", required=True, type=Path, help="held-out episodes (<logdir>/eval_eps) for the start states")
    ap.add_argument("--repo", type=Path, default=ROOT.parent / "dreamerv3-torch")
    ap.add_argument("--task", default="dmc_walker_walk")
    ap.add_argument("--min-return", type=float, default=600.0)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=60, help="imagined steps per start (20 steps = 1 s)")
    ap.add_argument("--true-episodes", type=int, default=4)
    ap.add_argument("--policy-set", choices=list(POLICY_SETS), default="coarse",
                    help="coarse: good to terrible play; fine: the actor with graded action noise, true returns close together")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "policy_ranking")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    repo = DreamerRepo(args.repo, task=args.task)
    base = DreamerWorldModel(repo, args.checkpoint)
    wm = base._load()
    actor = load_actor(repo, args.checkpoint, wm)
    specs = list(POLICY_SETS[args.policy_set])

    true_path = args.out_dir / "true_returns.json"
    settings = {"episodes": args.true_episodes, "policies": [s.name for s in specs]}
    if true_path.exists() and json.loads(true_path.read_text(encoding="utf-8")).get("settings") == settings:
        true = json.loads(true_path.read_text(encoding="utf-8"))["returns"]
        print("true returns: reused", true_path)
    else:
        t = time.perf_counter()
        true = true_returns(repo, wm, actor, specs, args.true_episodes)
        true_path.write_text(json.dumps({"settings": settings, "returns": true}, indent=1), encoding="utf-8")
        print(f"true returns: {len(specs)} policies x {args.true_episodes} episodes in {time.perf_counter() - t:.0f}s")
    true_mean = {k: st.mean(v) for k, v in true.items()}

    starts = load_starts(args.episodes, args.starts, min_return=args.min_return, seed=0)
    imagined: dict[str, dict[str, float]] = {}
    for name in args.configs:
        modules, kwargs, seed = CONFIGS[name]
        model = DreamerWorldModel(repo, args.checkpoint)
        stack = OptimizationStack(model, modules, module_kwargs=kwargs)
        if stack.skipped:
            print(f"{name}: skipped {[(s.name, s.reason) for s in stack.skipped]}")
            continue
        stack.apply()
        t = time.perf_counter()
        try:
            per_start = imagined_returns(model, actor, specs, starts, args.horizon, seed=seed)
        finally:
            stack.restore()
        imagined[name] = {k: st.mean(v) for k, v in per_start.items()}
        print(f"{name}: imagined returns in {time.perf_counter() - t:.0f}s", flush=True)
        del model, stack

    (args.out_dir / "imagined_returns.json").write_text(json.dumps(imagined, indent=1), encoding="utf-8")

    names = [s.name for s in specs]
    lines = [
        f"Policy returns on the Dreamer walker ({args.true_episodes} real episodes per policy; imagined over {args.horizon} steps from {len(starts)} held-out start states).",
        "",
        "| policy | true return | " + " | ".join(imagined) + " |",
        "|---|---|" + "---|" * len(imagined),
    ]
    for n in sorted(names, key=lambda n: -true_mean[n]):
        lines.append(f"| {n} | {true_mean[n]:.0f} | " + " | ".join(f"{imagined[c][n]:.1f}" for c in imagined) + " |")
    lines += ["", "Agreement of each model's ranking with the real simulator and with the unoptimized model's:", "",
              "| model | vs simulator | vs unoptimized model |", "|---|---|---|"]
    reference = imagined.get("baseline")
    for c, scores in imagined.items():
        vs_true = rank_agreement(true_mean, scores).summary()
        vs_base = "" if reference is None or c == "baseline" else rank_agreement(reference, scores).summary()
        lines.append(f"| {c} | {vs_true} | {vs_base} |")
    lines += ["", "Ranking, best first: simulator: " + " > ".join(ranking(true_mean))]
    for c, scores in imagined.items():
        lines.append(f"- {c}: " + " > ".join(ranking(scores)))
    text = "\n".join(lines)
    print("\n" + text)
    (args.out_dir / "report.md").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
