"""Run the Dreamer benchmark, optionally with an optimization stack applied.

    # baseline
    python scripts/run_dreamer.py --checkpoint ../dreamerv3-torch/logdir/walker --out results/dreamer_baseline.json

    # CUDA graphs (the launch-bound model's real bottleneck), vs the same baseline
    python scripts/run_dreamer.py --checkpoint ../dreamerv3-torch/logdir/walker         --cuda-graphs --baseline results/dreamer_baseline.json --out results/dreamer_graphs.json

    # quantized, with speedup measured against that baseline
    python scripts/run_dreamer.py --checkpoint ../dreamerv3-torch/logdir/walker         --quantize int8_weight_only --baseline results/dreamer_baseline.json         --out results/dreamer_int8.json

Don't time runs while something else is using the GPU (e.g. training): the
speed numbers are meaningless then, though the physics/visual numbers are fine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worldoptbench.models.dreamer import DreamerReplayScenarios, DreamerRepo, DreamerSimScenarios, DreamerWorldModel
from worldoptbench.reporting import summarize
from worldoptbench.runner import run_benchmark
from worldoptbench.stack import OptimizationStack

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SET = ROOT / "worldoptbench" / "prompts" / "dreamer_dmc_set.json"
DEFAULT_REPO = ROOT.parent / "dreamerv3-torch"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path, help="dreamerv3-torch logdir containing latest.pt")
    ap.add_argument("--out", required=True, type=Path, help="results JSON to write")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="dreamerv3-torch clone")
    ap.add_argument("--task", default="dmc_walker_walk")
    ap.add_argument("--standard-set", type=Path, default=DEFAULT_SET)
    ap.add_argument("--baseline", type=Path, help="earlier results JSON; speedup is measured against it")
    ap.add_argument("--episodes", type=Path, metavar="DIR",
                    help="score against held-out recorded episodes (e.g. <logdir>/eval_eps) instead of random-action simulator scenarios")
    ap.add_argument("--min-return", type=float, help="with --episodes: only use episodes whose return is at least this")
    ap.add_argument("--quantize", metavar="SCHEME", help="apply QuantizationModule with this scheme")
    ap.add_argument("--precision", choices=["bfloat16", "float16"], help="run under autocast in this dtype")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 float32 matmuls (process-wide)")
    ap.add_argument("--lean-scan", action="store_true", help="collect-and-stack imagination loop")
    ap.add_argument("--no-dist-validation", action="store_true", help="disable torch.distributions validation (process-wide)")
    ap.add_argument("--gumbel", action="store_true", help="pure-PyTorch Gumbel-max imagination step (reference for --tensorrt)")
    ap.add_argument("--tensorrt", action="store_true", help="run the RSSM step as a TensorRT engine (needs tensorrt-cu12 + onnx)")
    ap.add_argument("--cuda-graphs", action="store_true", help="replay the whole rollout as a CUDA graph")
    # experimental modules (built and unit-tested, not yet measured)
    ap.add_argument("--share-graph-pool", action="store_true", help="with --cuda-graphs: one shared memory pool for all graphs")
    ap.add_argument("--tensorrt-fp16", action="store_true", help="with --tensorrt: build the step engine in fp16")
    ap.add_argument("--tensorrt-decoder", action="store_true", help="run the image decoder as a TensorRT engine")
    ap.add_argument("--sparse-decode", type=int, metavar="STRIDE", help="decode every STRIDE-th frame and interpolate")
    ap.add_argument("--latent-noise", type=float, metavar="SCALE", help="scale latent sampling noise (0 = deterministic mode)")
    ap.add_argument("--channels-last", action="store_true", help="NHWC layout for conv weights")
    ap.add_argument("--cudnn-benchmark", action="store_true", help="cuDNN convolution autotuning (process-wide)")
    ap.add_argument("--low-rank", type=float, metavar="FRACTION", help="truncated-SVD factorization of Linear layers at this rank fraction")
    args = ap.parse_args()

    repo = DreamerRepo(args.repo, task=args.task)
    model = DreamerWorldModel(repo, args.checkpoint)

    modules, kwargs = [], {}
    if args.tf32:
        modules.append("tf32")
    if args.no_dist_validation:
        modules.append("no_dist_validation")
    if args.lean_scan:
        modules.append("lean_scan")
    if args.gumbel:
        modules.append("gumbel_sampling")
    if args.tensorrt:
        modules.append("tensorrt")
        if args.tensorrt_fp16:
            kwargs["tensorrt"] = {"precision": "fp16"}
    if args.tensorrt_decoder:
        modules.append("tensorrt_decoder")
    if args.sparse_decode:
        modules.append("sparse_decode")
        kwargs["sparse_decode"] = {"stride": args.sparse_decode}
    if args.latent_noise is not None:
        modules.append("latent_noise")
        kwargs["latent_noise"] = {"scale": args.latent_noise}
    if args.channels_last:
        modules.append("channels_last")
    if args.cudnn_benchmark:
        modules.append("cudnn_benchmark")
    if args.low_rank:
        modules.append("low_rank")  # before quantization: it can't factor an already-quantized layer
        kwargs["low_rank"] = {"rank_fraction": args.low_rank}
    if args.precision:
        modules.append("precision")
        kwargs["precision"] = {"dtype": args.precision}
    if args.quantize:
        modules.append("quantization")
        kwargs["quantization"] = {"scheme": args.quantize}
    if args.cuda_graphs:
        modules.append("cuda_graphs")  # after quantization: graphs are captured lazily, on the modified model
        if args.share_graph_pool:
            kwargs["cuda_graphs"] = {"share_pool": True}
    stack = OptimizationStack(model, modules, module_kwargs=kwargs)
    for skipped in stack.skipped:
        print(f"skipped: {skipped.reason}", file=sys.stderr)

    if args.episodes:
        longest = max(json.loads(args.standard_set.read_text(encoding="utf-8"))["horizons_seconds"])
        scenario_fn = DreamerReplayScenarios(
            args.episodes, repo.steps_per_second, max_horizon=longest, min_return=args.min_return
        )
        print(f"replay scenarios: {scenario_fn.num_episodes} episodes from {args.episodes}", file=sys.stderr)
    else:
        scenario_fn = DreamerSimScenarios(repo)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = run_benchmark(
        stack.apply(),
        standard_set_path=args.standard_set,
        output_path=args.out,
        baseline_path=args.baseline,
        optimization=stack.name,
        scenario_fn=scenario_fn,
    )
    print(f"{len(results)} rollouts -> {args.out}")
    print(summarize([r.__dict__ for r in results]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
