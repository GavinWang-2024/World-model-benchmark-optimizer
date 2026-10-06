"""Speed and memory scaling at long horizons (the 60 s and 120 s of outline sections 3.2 and 7.2) on the Dreamer model.

    python scripts/bench_long_horizon.py --checkpoint ../dreamerv3-torch/logdir/walker_long

What it measures and what it cannot: Dreamer's DMC walker episodes end at 500 agent steps (25 s), so the simulator cannot
provide ground truth beyond that and **no physics or drift score is computed here**: the drift curve stops at 20-25 s on
this task. What long horizons do test is whether the cost grows linearly with the rollout length, how much GPU memory the
captured graphs hold, and whether the model returns every frame. Inputs are synthetic (zero context frames, random actions),
timings are medians of --repeats calls after warm-up, and each configuration is a fresh model.

Configurations: the unoptimized eager rollout, and the best measured stack (TensorRT fp16 + CUDA graphs).
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
    "eager (unoptimized)": ([], {}),
    "tensorrt fp16 + cuda_graphs": (["tensorrt", "cuda_graphs"], {"tensorrt": {"precision": "fp16"}}),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=ROOT.parent / "dreamerv3-torch")
    ap.add_argument("--horizons", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0, 60.0, 120.0])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    import torch

    repo = DreamerRepo(args.repo, task="dmc_walker_walk")
    n_actions = repo.config.num_actions
    rng = np.random.default_rng(0)
    context = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
    context_actions = [rng.uniform(-1, 1, n_actions).astype(np.float32) for _ in range(4)]

    print(f"median latency over {args.repeats} calls; frames returned; GPU memory held after warm-up")
    for label, (names, kwargs) in CONFIGS.items():
        model = DreamerWorldModel(repo, args.checkpoint)
        model._load()
        stack = OptimizationStack(model, names, module_kwargs=kwargs)
        for skipped in stack.skipped:
            raise SystemExit(f"skipped {skipped.name}: {skipped.reason}")
        stack.apply()
        print(f"\n{label}")
        print(f"{'horizon':>9s} {'steps':>6s} {'frames':>7s} {'latency ms':>11s} {'ms / sim second':>16s} {'held MB':>8s}")
        for horizon in args.horizons:
            steps = horizon_to_steps(horizon, repo.steps_per_second)
            actions = [rng.uniform(-1, 1, n_actions).astype(np.float32) for _ in range(steps)]
            call = lambda: model.generate(init_video=context, actions=actions, horizon=horizon, context_actions=context_actions, seed=3)
            for _ in range(2):
                out = call()
            times = []
            for _ in range(args.repeats):
                t = time.perf_counter()
                out = call()
                times.append((time.perf_counter() - t) * 1000)
            median = st.median(times)
            print(f"{horizon:8g}s {steps:6d} {len(out.frames):7d} {median:11.1f} {median / horizon:16.2f} {torch.cuda.memory_allocated() / 2**20:8.0f}")
            assert len(out.frames) == steps, "the model must return one frame per step"
        stack.restore()
        del model, stack
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
