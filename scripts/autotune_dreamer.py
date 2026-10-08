"""Let the library choose a stack for the Dreamer model, by measurement.

    python scripts/autotune_dreamer.py --checkpoint ../dreamerv3-torch/logdir/walker_long \\
        --standard-set worldoptbench/prompts/dreamer_dmc_set.json --repeats 2

Greedy forward selection (worldoptbench.autotune): start from no optimization, add
whichever applicable module most improves PAES, stop when nothing helps by more than
--min-gain, and reject any stack that loses more than --max-physics-drop of physics.

This has been unit-tested against scripted scores but never run end to end on
hardware — treat the first run as a test of autotune itself, not just of the model.
Cost: every candidate stack is a full benchmark (x --repeats), so the first rounds
are slow (tens of minutes); --candidates restricts the search, --repeats 1 is fast
but noisy (the eager baseline varies +-34% run to run), and the keep-the-GPU-idle
rule applies here too. Scenarios are built once and reused across every evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldoptbench.autotune import autotune, default_candidates, evaluate_with_runner
from worldoptbench.models.dreamer import (
    DreamerReplayScenarios,
    DreamerRepo,
    DreamerSimScenarios,
    DreamerWorldModel,
)
from worldoptbench.optimizations.base import get_module_class
from worldoptbench.scenarios import Scenario

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SET = ROOT / "worldoptbench" / "prompts" / "dreamer_dmc_set.json"
DEFAULT_REPO = ROOT.parent / "dreamerv3-torch"


def _load_kwargs(value: str) -> dict:
    text = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    return json.loads(text)


class _CachedScenarios:
    """Scenarios are deterministic per (entry, horizon), so build each simulator rollout
    once instead of once per candidate stack."""

    def __init__(self, inner):
        self._inner = inner
        self._cache: dict[tuple[str, float], Scenario] = {}

    def __call__(self, entry, horizon):
        key = (entry["id"], horizon)
        if key not in self._cache:
            self._cache[key] = self._inner(entry, horizon)
        return self._cache[key]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--task", default="dmc_walker_walk")
    ap.add_argument("--standard-set", type=Path, default=DEFAULT_SET)
    ap.add_argument("--episodes", type=Path, metavar="DIR", help="score against held-out recorded episodes (e.g. <logdir>/eval_eps)")
    ap.add_argument("--min-return", type=float, help="with --episodes: only episodes with at least this return")
    ap.add_argument("--module-kwargs", type=_load_kwargs, default=None,
                    help='JSON {module: {kwarg: value}}, or @path to a JSON file (easier to quote on Windows), '
                         'e.g. {"tensorrt": {"precision": "fp16"}}')
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "autotune.json")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=0.10, help="relative PAES gain a module must add (0.10 = 10%%)")
    ap.add_argument("--max-physics-drop", type=float, default=0.02)
    ap.add_argument("--max-modules", type=int, default=6)
    ap.add_argument("--candidates", nargs="*", help="module names to consider (default: every applicable module)")
    ap.add_argument("--measured-only", action="store_true", help="skip experimental modules")
    args = ap.parse_args()

    repo = DreamerRepo(args.repo, task=args.task)

    def factory():
        return DreamerWorldModel(repo, args.checkpoint)

    candidates = args.candidates or default_candidates(factory())
    if args.measured_only:
        candidates = [c for c in candidates if get_module_class(c).maturity == "measured"]
    print(f"candidates ({len(candidates)}): {', '.join(candidates)}", flush=True)

    if args.episodes:
        longest = max(json.loads(args.standard_set.read_text(encoding="utf-8"))["horizons_seconds"])
        provider = DreamerReplayScenarios(
            args.episodes, repo.steps_per_second, max_horizon=longest, min_return=args.min_return
        )
        print(f"replay scenarios: {provider.num_episodes} episodes", flush=True)
    else:
        provider = DreamerSimScenarios(repo)
    evaluate = evaluate_with_runner(
        factory,
        {"standard_set_path": args.standard_set, "scenario_fn": _CachedScenarios(provider)},
        repeats=args.repeats,
        module_kwargs=args.module_kwargs,
    )
    result = autotune(
        candidates, evaluate,
        min_gain=args.min_gain, max_physics_drop=args.max_physics_drop, max_modules=args.max_modules,
    )

    print("\nsteps tried:")
    for step in result.trace:
        mark = "*" if step.accepted else " "
        print(f" {mark} {'+'.join(step.modules):55s} PAES {step.evaluation.paes:7.3f}  physics {step.evaluation.physics:.3f}  {step.note}")
    print(f"\nbaseline: PAES {result.baseline.paes:.3f}, physics {result.baseline.physics:.3f}")
    print(f"best stack: {result.best_modules or '(none: nothing helped by more than the margin)'}")
    print(f"           PAES {result.best.paes:.3f}, physics {result.best.physics:.3f}, mean latency "
          f"{1000 * (result.best.latency_seconds or 0):.1f} ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "best_modules": result.best_modules,
        "best": result.best.__dict__, "baseline": result.baseline.__dict__,
        "trace": [{"modules": t.modules, "paes": t.evaluation.paes, "physics": t.evaluation.physics,
                   "accepted": t.accepted, "note": t.note} for t in result.trace],
    }, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
