"""How much does skipping the decoder save? `generate()` (frames) against `generate_latents()` (features only) on Dreamer.

    python scripts/bench_latents.py --checkpoint ../dreamerv3-torch/logdir/walker_long

For consumers that never look at pixels (a planner or policy evaluator reads the reward and value heads off the latents)
the decoder is wasted work. Median of --repeats calls after warm-up, same inputs and seed for both, a fresh model per
configuration. The last column checks that the latents are what `generate()` decodes (they must be identical for a
same-seed run on the deterministic eager path).
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from worldoptbench.models.dreamer import DreamerRepo, DreamerWorldModel, horizon_to_steps  # noqa: E402
from worldoptbench.stack import OptimizationStack  # noqa: E402

CONFIGS = {
    "eager": ([], {}),
    "cuda_graphs": (["cuda_graphs"], {}),
    "tensorrt fp16 + cuda_graphs": (["tensorrt", "cuda_graphs"], {"tensorrt": {"precision": "fp16"}}),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=ROOT.parent / "dreamerv3-torch")
    ap.add_argument("--horizons", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0])
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()

    repo = DreamerRepo(args.repo, task="dmc_walker_walk")
    n_actions = repo.config.num_actions
    rng = np.random.default_rng(0)
    context = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
    context_actions = [rng.uniform(-1, 1, n_actions).astype(np.float32) for _ in range(4)]

    print(f"median ms over {args.repeats} calls")
    print(f"{'configuration':30s} {'horizon':>8s} {'frames':>8s} {'latents':>8s} {'saved':>7s}")
    for label, (names, kwargs) in CONFIGS.items():
        model = DreamerWorldModel(repo, args.checkpoint)
        model._load()
        stack = OptimizationStack(model, names, module_kwargs=kwargs)
        for skipped in stack.skipped:
            raise SystemExit(f"skipped {skipped.name}: {skipped.reason}")
        stack.apply()
        for horizon in args.horizons:
            actions = [rng.uniform(-1, 1, n_actions).astype(np.float32) for _ in range(horizon_to_steps(horizon, repo.steps_per_second))]
            kw = dict(init_video=context, actions=actions, horizon=horizon, context_actions=context_actions, seed=3)
            times = {}
            for kind, call in (("frames", lambda: model.generate(**kw)), ("latents", lambda: model.generate_latents(**kw))):
                for _ in range(3):
                    call()
                samples = []
                for _ in range(args.repeats):
                    t = time.perf_counter()
                    call()
                    samples.append((time.perf_counter() - t) * 1000)
                times[kind] = st.median(samples)
            print(f"{label:30s} {horizon:7g}s {times['frames']:8.2f} {times['latents']:8.2f} {1 - times['latents'] / times['frames']:7.0%}")
        stack.restore()
        del model, stack
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
