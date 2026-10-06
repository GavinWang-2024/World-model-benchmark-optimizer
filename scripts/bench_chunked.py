"""Chunked vs whole-rollout CUDA graphs on the Dreamer model: equivalence and cost of the chunk boundaries.

    python scripts/bench_chunked.py --checkpoint ../dreamerv3-torch/logdir/walker_long

For each configuration (a fresh model each) it runs the same inputs and seed at several horizons and reports
the median latency per horizon, how far the frames are from the whole-rollout run (max absolute pixel
difference, and the fraction of pixels that differ), the number of captured graphs and the memory held
after warm-up. Keep the GPU otherwise idle.

DESIGN_STEP_HOOK.md asks two questions before chunking is worth keeping: does it reproduce the whole-rollout
output (same seed), and is the boundary overhead small enough?
"""

from __future__ import annotations

import argparse
import statistics as st
import time
from pathlib import Path

import numpy as np

from worldoptbench.models.dreamer import DreamerRepo, DreamerWorldModel, horizon_to_steps
from worldoptbench.stack import OptimizationStack

ROOT = Path(__file__).resolve().parent.parent


def build(repo, checkpoint, stack_names, kwargs):
    model = DreamerWorldModel(repo, checkpoint)
    model._load()
    stack = OptimizationStack(model, stack_names, module_kwargs=kwargs)
    for skipped in stack.skipped:
        raise SystemExit(f"skipped {skipped.name}: {skipped.reason}")
    return stack.apply(), stack


def request(repo, horizon, rng):
    n_actions = repo.config.num_actions
    steps = horizon_to_steps(horizon, repo.steps_per_second)
    return {
        "init_video": [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)],
        "context_actions": [rng.uniform(-1, 1, n_actions).astype(np.float32) for _ in range(4)],
        "actions": [rng.uniform(-1, 1, n_actions).astype(np.float32) for _ in range(steps)],
        "horizon": horizon,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=ROOT.parent / "dreamerv3-torch")
    ap.add_argument("--horizons", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0, 20.0])
    ap.add_argument("--chunks", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--backend", choices=["tensorrt_fp16", "gumbel"], default="tensorrt_fp16")
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()

    import torch

    repo = DreamerRepo(args.repo, task="dmc_walker_walk")
    if args.backend == "tensorrt_fp16":
        backend_names, backend_kwargs = ["tensorrt"], {"tensorrt": {"precision": "fp16"}}
    else:
        backend_names, backend_kwargs = ["gumbel_sampling"], {}

    configs = [("whole rollout (one graph per horizon)", backend_names + ["cuda_graphs"], dict(backend_kwargs), None)]
    for k in args.chunks:
        configs.append((f"chunked K={k}", backend_names + ["chunked_rollout", "cuda_graphs"],
                        {**backend_kwargs, "chunked_rollout": {"chunk_steps": k}}, k))

    rng = np.random.default_rng(0)
    requests = {h: request(repo, h, rng) for h in args.horizons}
    reference_frames: dict[float, np.ndarray] = {}

    print(f"backend {args.backend}; median latency in ms over {args.repeats} calls; diff vs the whole-rollout frames")
    print(f"{'configuration':38s} " + " ".join(f"{h:>7g}s" for h in args.horizons) + "   graphs  held MB   max|diff|  differing px")
    for label, names, kwargs, _ in configs:
        model, stack = build(repo, args.checkpoint, names, kwargs)
        medians, worst, differing = [], 0, 0.0
        for h in args.horizons:
            req = requests[h]
            for _ in range(3):  # capture + warm-up
                model.generate(init_video=req["init_video"], actions=req["actions"], horizon=h,
                               context_actions=req["context_actions"], seed=7)
            times = []
            for _ in range(args.repeats):
                t = time.perf_counter()
                out = model.generate(init_video=req["init_video"], actions=req["actions"], horizon=h,
                                     context_actions=req["context_actions"], seed=7)
                times.append((time.perf_counter() - t) * 1000)
            medians.append(st.median(times))
            frames = np.stack(out.frames).astype(np.int16)
            if label.startswith("whole"):
                reference_frames[h] = frames
            else:
                delta = np.abs(frames - reference_frames[h])
                worst = max(worst, int(delta.max()))
                differing = max(differing, float((delta > 0).mean()))
        executor = model.tensor_executor
        graphs = getattr(executor, "num_graphs", 0)
        held = torch.cuda.memory_allocated() / 2**20
        diff = "" if label.startswith("whole") else f"{worst:9d} {differing:12.4%}"
        print(f"{label:38s} " + " ".join(f"{m:8.1f}" for m in medians) + f"   {graphs:6d} {held:8.0f}   {diff}")
        stack.restore()
        del model, stack, executor
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
