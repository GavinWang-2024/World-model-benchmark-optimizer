# WorldOptBench

Architecture-agnostic benchmarking and optimization framework for world model inference — joint speed, visual quality, and physics consistency (see `PAES`).

- **Full spec / pitch doc:** [`world_model_project_outline.md`](world_model_project_outline.md)
- **Execution checklist (what to build in what order):** [`build_plan.md`](build_plan.md)
- **Every optimization we could add (125, with status):** [`OPTIMIZATION_CATALOG.md`](OPTIMIZATION_CATALOG.md)

## Setup

```bash
pip install -e ".[dev]"
pytest
```

GPU/model work needs the `ml` extra and a hardware-appropriate Torch/CUDA build — install Torch yourself first per your rented/HPC box, then:

```bash
pip install -e ".[ml]"
```

See `build_plan.md` Phase 0 for HF access + compute setup.

## Running Dreamer (no Hugging Face needed)

Train a checkpoint with [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch), then:

```python
from worldoptbench.models.dreamer import DreamerRepo, DreamerWorldModel, DreamerSimScenarios
from worldoptbench.runner import run_benchmark

repo = DreamerRepo("path/to/dreamerv3-torch", task="dmc_walker_walk")
results = run_benchmark(
    DreamerWorldModel(repo, "path/to/dreamerv3-torch/logdir/walker"),
    standard_set_path="worldoptbench/prompts/dreamer_dmc_set.json",
    scenario_fn=DreamerSimScenarios(repo),
    output_path="baseline_dreamer.json",
)
```

Then the same run with quantization, compared against the baseline (use a fresh model for each run — stacks modify the model in place):

```python
from worldoptbench.stack import OptimizationStack

stack = OptimizationStack(
    DreamerWorldModel(repo, "path/to/dreamerv3-torch/logdir/walker"),
    ["quantization"],
    module_kwargs={"quantization": {"scheme": "int8_weight_only"}},
)
results = run_benchmark(
    stack.apply(),
    optimization=stack.name,
    baseline_path="baseline_dreamer.json",  # speedup is measured against this
    standard_set_path="worldoptbench/prompts/dreamer_dmc_set.json",
    scenario_fn=DreamerSimScenarios(repo),
    output_path="quantized_dreamer.json",
)
```

The Dreamer model is launch-bound (a Python loop of tiny kernels), so the optimizations that actually help are CUDA graphs (~9x) and TensorRT on the RSSM step (a further ~1.6x), not quantization: `scripts/run_dreamer.py --tensorrt --cuda-graphs` (needs `tensorrt-cu12` and `onnx`; see `DREAMER_SETUP.md` for the full results table and caveats).

On Windows, set `MUJOCO_GL=glfw` (or `egl` on Linux GPU boxes) if rendering fails. Test commands: [`TEST_COMMANDS.md`](TEST_COMMANDS.md).

## Diffusion models (Wan2.1) and the quality caveat

`worldoptbench/models/wan_video.py` wraps Wan2.1-T2V-1.3B (public, runs on a 12 GB GPU). Encode prompts once on CPU
(`python scripts/encode_prompts.py --from-set worldoptbench/prompts/wan_set_16.json`), then sweep optimizations with
`scripts/sweep_wan.py` (one process per configuration; `scripts/analyze_blind.py` and `scripts/compare_diffusion.py`
summarize). Results, the noise floor of the PSNR metric, and how to read them are in
[`DESIGN_DIFFUSION.md`](DESIGN_DIFFUSION.md). The library has the diffusers caches, CFG truncation, and our own
WorldCache and AdaCache implementations; `defaults.recommended_config(model)` picks a measured stack.

## Serving: worldserve

```bash
worldserve --model wan --stack recommended --port 8000          # or: python -m worldoptbench.serve ...
curl -s -X POST localhost:8000/generate -d '{"prompt": "A red ball rolls down a ramp.", "horizon": 2, "seed": 0}'
curl -s localhost:8000/info ; curl -s localhost:8000/profile
```

Standard-library HTTP server in front of one model plus an optimization stack: bounded request queue (429 when full),
micro-batching for models with `generate_batch`, an optional result cache, latency/queue profile. Frames come back as
base64 PNGs. No auth or TLS (put a proxy in front). The `Dockerfile` is untested. Details in `worldoptbench/serve.py`.

## Measuring and choosing

- **Profile of a run** (the block in the outline's section 3.2): `from worldoptbench.reporting import format_profile`, then
  `print(format_profile(json.load(open("results.json"))))`: model, architecture, optimization, hardware, speed (including
  GPU memory held), visual (PSNR, temporal consistency, and DINO similarity / style deviation when computed), physics, drift and PAES.
- **Constraints**: `Constraints(min_physics_score=..., target_speedup=..., max_vram_gb=..., max_latency_seconds=...)`; check a measured run with
  `OptimizationStack(model, modules, constraints=...).check(results)`, or search under them with `autotune(..., constraints=...)`.
  Constraints are checked against measured numbers, never predicted.
- **Pareto plots and tables**: `scripts/plot_pareto.py` (needs `pip install matplotlib`), `scripts/compare_diffusion.py`,
  `scripts/analyze_perceptual.py`, `scripts/analyze_fvd.py` (a DINO-space Frechet distance; not standard FVD).
- **Cost**: `scripts/cost_pareto.py results/wan_perceptual --usd-per-hour <rate>`: dollars per generated second and the cheapest
  configuration above a quality floor. The rate must be the cost of the machine the latencies were measured on.
- **Leaderboard**: `python scripts/make_leaderboard.py` writes `LEADERBOARD.md` from the result files.
- **Does an optimization preserve what the model is used for?** `scripts/policy_ranking_dreamer.py` ranks a set of policies in the
  real simulator and inside each optimized Dreamer world model and compares the rankings (see `OUTLINE_STATUS.md`).
