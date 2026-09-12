# WorldOptBench — Project Outline

> **One-liner:** A general framework for benchmarking and optimizing world model inference across architectures — treating speed, visual quality, and physics consistency as a joint objective for the first time.

---

## 1. The Problem

World model inference is **8–32x more expensive** than LLM inference. A wave of optimization papers has emerged in 2026:

| Paper | Technique | Speedup | Architecture | Physics Eval? |
|-------|-----------|---------|--------------|---------------|
| WorldCache (Mar 2026) | Heterogeneous token caching | 2.3x | Diffusion only | ❌ |
| AdaCache | Adaptive step caching | 4.7x | Diffusion only | ❌ |
| AccVideo | Distillation | 8.5x | Diffusion only | ❌ |
| OpenWorldLib (Apr 2026) | Unified inference codebase | — | Mixed | ❌ |

**Every single one optimizes for speed + visual quality only — and most are diffusion-specific.**

World models are not just diffusion models. The field includes autoregressive models (Genie), JEPA-style latent models (V-JEPA 2), and diffusion models (Cosmos-Predict) — all with the same core problems: too slow, no physics guarantees, no standard way to measure tradeoffs across architectures.

For the actual customers — robotics teams, AV simulation, game studios — **physics consistency and long-horizon stability matter as much as speed**, regardless of what architecture is underneath.

**Confirmed open problems from the literature:**
- "The field still lacks unified, interpretable, and closed-loop-relevant evaluation protocols for long-horizon stability" — *Latent World Models for Automated Driving, 2026*
- "Maintaining coherent world evolution over extremely long horizons remains one of the most critical open problems" — *Towards Interactive Video World Modeling, 2026*
- WorldRoamBench (Jun 2026) evaluated 10+ models across architectures — none reliably satisfies all dimensions

---

## 2. What We're Building

**WorldOptBench** — a general, architecture-agnostic framework with two components:

```
WorldOptBench
├── Benchmark Suite        ← standardized eval across speed + visual + physics
│                            works across diffusion, autoregressive, JEPA models
└── Optimization Stack     ← plug-in modules, architecture-aware
    └── Serving Layer      ← production inference server (worldserve)
```

The thesis: **if you can't measure all three axes jointly across architectures, you can't optimize them jointly.** We build the measurement standard first, then the optimization stack on top of it.

The generality lives at the interface layer — each model implements a standard `WorldModelInterface`, the benchmark doesn't care what's underneath. Optimization modules are architecture-specific where they have to be (diffusion caching only works for diffusion), but the framework that wraps them is universal.

---

## 3. Core Features (v1 — What We Actually Ship)

These are the non-negotiable deliverables for a first version. Everything else is future work.

### 3.1 WorldModelInterface (The Abstraction Layer)

A standard interface any world model implements so the benchmark and optimization stack don't care about the underlying architecture:

```python
class WorldModelInterface:
    """
    Any world model — diffusion, autoregressive, JEPA — implements this.
    The benchmark runner only calls .generate() and .get_info().
    """
    def generate(
        self,
        prompt: str,
        actions: list,       # action sequence conditioning
        horizon: float,      # rollout length in seconds
        **kwargs,
    ) -> list:               # returns list of frames
        raise NotImplementedError

    def get_info(self) -> dict:
        # returns architecture type, model size, supported optimizations
        raise NotImplementedError
```

**v1 implementations:**
| Model | Architecture | Interface Implemented |
|-------|-------------|----------------------|
| Cosmos-Predict 7B | Diffusion (DiT) | ✅ Primary target |
| Cosmos-Predict 2.5-2B | Diffusion (DiT) | ✅ Lightweight experiments |
| Open-Sora | Diffusion (DiT) | ✅ Secondary target |
| Genie 3 | Autoregressive | ⏳ Future (API closed) |
| V-JEPA 2 | JEPA latent | ⏳ Future work |

---

### 3.2 Unified Benchmark Suite

A single eval harness that runs any optimization technique against any world model implementing the interface, and outputs a standardized profile across three axes:

**Axis 1 — Speed**
- Wall-clock latency (seconds per generated second of video)
- Frames per second throughput
- GPU memory peak usage
- Time to first frame

**Axis 2 — Visual Quality**
- FVD (Fréchet Video Distance)
- PSNR / SSIM per frame
- Temporal consistency (frame-to-frame delta)

**Axis 3 — Physics Consistency** ← the new one
- PAI-Bench score (physical plausibility + controllability)
- WorldRoamBench physics subscore
- Long-horizon drift curve — physics score plotted against rollout length at 4s / 16s / 60s / 120s

**Output for every run:**
```
Model:        Cosmos-Predict 7B
Architecture: Diffusion (DiT)
Optimization: WorldCache
Hardware:     H100

Speed:        2.3x speedup | 18.4s/s generation | 42GB VRAM peak
Visual:       FVD 84.2 | PSNR 31.1 | Temporal consistency 0.94
Physics:      PAI-Bench 0.71 | Drift onset: 28s | Drift rate: -0.008/s

PAES Score:   1.84
```

---

### 3.3 Physics-Aware Efficiency Score (PAES)

A single combined metric — architecture agnostic — so any optimization paper has one number to compare against:

```
PAES = Speedup × Physics_Score × (1 / Drift_Penalty)

where Drift_Penalty = 1 + |drift_rate| × horizon_T
```

- `Speedup` = baseline latency / optimized latency
- `Physics_Score` = PAI-Bench or WorldRoamBench physics subscore (0–1)
- `Drift_Penalty` = how much physics degrades over the rollout horizon

**Why this matters:** A method that gets 4x speedup with 50% physics degradation should score lower than one that gets 2x speedup with 5% degradation. No current metric captures this. PAES does — and it works identically whether the model is diffusion, autoregressive, or JEPA.

---

### 3.4 Optimization Stack (Plug-in Architecture)

A modular system where optimization techniques are composable components. The framework is architecture-agnostic; individual modules declare which architectures they support:

```python
from worldoptbench import OptimizationStack

stack = OptimizationStack(
    model="nvidia/cosmos-predict-7b",   # implements WorldModelInterface
    modules=[
        "worldcache",        # diffusion only — auto-skipped for AR models
        "adacache",          # diffusion only
        "fp8_quantization",  # all architectures
        "drift_correction",  # diffusion only (our novel piece, v1)
        "kv_cache_ar",       # autoregressive only — future module
    ],
    constraints={
        "min_physics_score": 0.85,
        "target_speedup": 2.0,
        "max_vram_gb": 80,
    }
)
```

The stack auto-filters incompatible modules based on the model's architecture type. You can use the same API regardless of what model you're running.

---

### 3.5 Inference-Time Drift Correction (Novel Research Piece)

The one technically novel contribution — a training-free module for diffusion world models that preserves physics consistency under caching optimizations.

**The mechanism:**
1. **Monitor** optical flow divergence across generated frames as a lightweight physics drift proxy
2. **Detect** when the signal crosses a threshold — drift onset
3. **Correct** by selectively recomputing only the high-curvature tokens (WorldCache already scores these), not the full frame
4. **Blend** corrections back into the cached state

Runs on top of WorldCache as a composable module. No retraining. No new weights.

**Target:** Match WorldCache's 2.3x speedup while recovering 60–80% of the physics consistency gap vs. unoptimized baseline. A clear Pareto improvement on PAES.

**Fallback:** If this doesn't pan out, the benchmark + PAES metric + architecture-agnostic framework is already a publishable contribution on its own.

---

### 3.6 `worldserve` Serving Layer

Production inference server wrapping the optimization stack. The gap vs. OpenWorldLib: OpenWorldLib is a research codebase for calling models. `worldserve` is a server you actually deploy, with any architecture underneath.

```python
pip install worldserve

from worldserve import WorldModelServer

server = WorldModelServer(
    model="nvidia/cosmos-predict-7b",    # or any WorldModelInterface implementation
    stack=["worldcache", "drift_correction"],
    hardware="H100",
    physics_budget=0.90,
    speed_budget=2.0,
)

server.serve(port=8000)   # HTTP endpoint

# or direct Python
frames = server.generate(
    prompt="robot arm picking up red cube",
    horizon=60,
    action_sequence=[...],
)
```

**v1 features:**
- Architecture-agnostic — same server works for diffusion and AR models
- Works on single H100 (480p) or RTX 4090 with quantization (360p)
- FastAPI + async batching
- Returns PAES profile alongside every generation
- Docker container for easy deployment

---

## 4. Future Work (Post-v1)

Things explicitly out of scope for v1 but natural extensions:

- **More model implementations:** Genie 3 (API closed currently), V-JEPA 2, WAN2.1, custom fine-tunes
- **AR-specific optimization modules:** KV caching for autoregressive world models, speculative decoding (SDVG)
- **JEPA-specific modules:** Latent space optimization for JEPA-style architectures
- **Pipeline parallelism:** Different GPUs handle different model layers — more complex than tensor parallel, higher throughput for very large models
- **Spatial parallelism:** Split video frames spatially across GPUs (GPU 1 handles left half, GPU 2 handles right half) — novel for video generation, used by Video Infinity
- **Disaggregated prefill/decode:** Separate GPUs for prefill vs. decode phases, same technique Baseten uses for LLMs
- **Domain-specific physics metrics:** Robotics (contact dynamics, grasp stability), AV (trajectory plausibility)
- **Managed API product:** If the library gets traction, productize as a hosted endpoint
- **Distributed training / fine-tuning:** Allow domain-specific fine-tuning of world models across multiple GPUs using DeepSpeed or FSDP — v2 product feature for robotics companies
- **LLM optimization sidebar:** Dynamic depth routing, attention residuals — lower priority

---

## 5. Related Work & How We Differ

| Project | What It Does | Gap |
|---------|-------------|-----|
| WorldCache | Diffusion caching, 2.3x speedup | Diffusion only, no physics eval |
| AdaCache | Step-skipping, 4.7x speedup | Diffusion only, no physics eval |
| OpenWorldLib | Unified inference codebase | No optimization, no physics eval, not arch-agnostic |
| WorldRoamBench | Long-horizon benchmark | Benchmark only, no optimization |
| PAI-Bench | Physics plausibility eval | No speed component, no arch comparison |
| **WorldOptBench (ours)** | **Architecture-agnostic framework: joint optimization + benchmarking across all three axes** | — |

---

## 6. General Technologies

**Core language:** Python 3.11+

**ML / Model layer:**
- PyTorch — tensor operations, hooking into model internals for drift correction
- HuggingFace Diffusers — base library for diffusion models (Cosmos-Predict, Open-Sora)
- HuggingFace Hub — model weight downloads
- HuggingFace Transformers — for autoregressive model support (future)
- xFormers / Flash Attention 2 — memory-efficient attention

**Optimization modules:**
- WorldCache / AdaCache — pulled from open-source repos, wrapped as plug-in modules
- BitsAndBytes / TorchAO — FP8 and INT4 quantization (architecture agnostic)
- CUDA — assumed as default runtime, no explicit custom kernels in v1
- Triton — only if custom kernels needed (not in v1, future work)

**Distributed inference (v1):**
- HuggingFace Accelerate — handles tensor parallelism across multiple GPUs with minimal code changes; splits model weight matrices across GPUs so larger models (7B at 720p) fit in available VRAM
- NCCL — NVIDIA's GPU communication library, used under the hood by Accelerate for inter-GPU communication; you don't write to this directly
- `torch.distributed` — PyTorch's built-in distributed primitives, used by Accelerate internally

```python
# How it surfaces in worldserve
server = WorldModelServer(
    model="nvidia/cosmos-predict-7b",
    distributed={
        "strategy": "tensor_parallel",  # splits weights across GPUs
        "num_gpus": 4,
    },
    stack=["worldcache", "drift_correction"],
)
```

**CUDA memory management (v1):**
- Automatic dtype selection based on available VRAM — bfloat16 for H100, float16 for A100, int8 for consumer GPUs
- Automatic VRAM check on startup — warns if model won't fit, suggests quantization or distributed config
- Peak VRAM tracking per benchmark run via PyTorch memory profiler

**Benchmarking:**
- PyTorch profiler + `nvitop` / `nvidia-smi` — GPU memory and utilization
- `torchmetrics` — FVD, PSNR, SSIM
- PAI-Bench + WorldRoamBench — physics scoring (pulled from public repos)
- NumPy / pandas — storing and processing results
- Matplotlib / seaborn — Pareto plots, drift curves for the paper

**Serving layer:**
- FastAPI — HTTP server
- Uvicorn — ASGI server
- Pydantic — request/response validation
- Docker — containerized deployment

**Infrastructure:**
- RunPod / Lambda Labs — rented H100s for experiments
- GitHub Actions — CI
- pytest — test suite

---

## 7. Implementation Approach — Feature by Feature

### Build Order (Do These In This Sequence)

```
1. WorldModelInterface + Cosmos-Predict implementation
2. Benchmark runner (speed + visual metrics)
3. Physics metrics (PAI-Bench, WorldRoamBench)
4. WorldCache wrapper module
5. PAES metric
6. AdaCache + quantization modules
7. Speculative decoding module
8. Distributed inference + CUDA memory management
9. Drift correction module  ← hardest
10. worldserve server       ← last
```

---

### 7.1 WorldModelInterface + First Implementation

**What it is:** Define the abstraction, then implement it for Cosmos-Predict. Everything else builds on top of this.

**Steps:**
1. Define the `WorldModelInterface` base class in `worldoptbench/models/base.py`
2. Implement `CosmosPredict2B` and `CosmosPredict7B` classes that wrap HuggingFace Diffusers and implement the interface
3. Verify: call `.generate()` on both, get frames out, confirm it runs on your hardware

**What you're coding:**
```
worldoptbench/
└── models/
    ├── base.py              ← WorldModelInterface abstract class
    ├── cosmos_predict.py    ← Cosmos-Predict 2B + 7B implementations
    └── open_sora.py         ← Open-Sora implementation (secondary)
```

**Hardest part:** Cosmos-Predict is gated on HuggingFace — need to request access. Also getting CUDA dependencies right for your hardware. Expect a day of setup friction.

---

### 7.2 Benchmark Runner (Speed + Visual Metrics)

**What it is:** A script that takes any `WorldModelInterface` implementation, runs a fixed set of rollouts, and outputs a structured results JSON.

**Steps:**
1. Define the standard rollout set — fixed prompts + action sequences at 4s / 16s / 60s / 120s horizons
2. Write the runner — calls `.generate()`, wraps it with PyTorch profiler to capture wall-clock time and VRAM
3. Compute visual metrics on output frames — FVD, PSNR, SSIM using `torchmetrics`
4. Write results to JSON
5. Write a comparison script that takes multiple JSONs and produces Pareto plots

**What you're coding:**
```
worldoptbench/
├── runner.py                 ← orchestrates benchmark runs
├── metrics/
│   ├── speed.py              ← latency, throughput, VRAM
│   └── visual.py             ← FVD, PSNR, SSIM
└── prompts/
    └── standard_set.json     ← fixed rollout set
```

---

### 7.3 Physics Metrics

**What it is:** Wrappers around PAI-Bench and WorldRoamBench that take generated frames and return physics scores.

**Steps:**
1. Clone PAI-Bench and WorldRoamBench repos
2. Get them running standalone — both are research codebases, expect rough edges
3. Wrap them behind a clean `physics.py` interface that takes frames → returns score dict
4. Add physics scoring to the benchmark runner output

**What you're coding:**
```
worldoptbench/metrics/
└── physics.py    ← PAI-Bench + WorldRoamBench wrappers
```

**Hardest part:** Both are research codebases with rough dependency management. Getting them to run cleanly is the main challenge. Do this early so you hit problems early.

---

### 7.4 Optimization Stack + WorldCache Module

**What it is:** The plug-in architecture base class, plus the first real optimization module.

**Steps:**
1. Define `OptimizationModule` base class — `apply(pipeline, config) → pipeline`, plus `supported_architectures` field
2. Write `OptimizationStack` — takes a list of modules, filters incompatible ones based on model architecture, chains compatible ones
3. Wrap WorldCache as the first module — clone their repo, import core caching logic, adapt to interface
4. Run the benchmark with and without WorldCache to confirm you reproduced their 2.3x result

**What you're coding:**
```
worldoptbench/
├── optimizations/
│   ├── base.py                  ← OptimizationModule abstract class
│   ├── worldcache.py            ← WorldCache wrapper
│   ├── adacache.py              ← AdaCache wrapper (added in 7.6)
│   ├── quantization.py          ← FP8/INT4 via TorchAO (added in 7.6)
│   ├── speculative_decoding.py  ← SDVG-style speculative decoding (added in 7.7)
│   └── drift_correction.py      ← our novel module (added in 7.9)
└── stack.py                     ← OptimizationStack + constraint solver
```

**Hardest part:** WorldCache hooks into HuggingFace Diffusers' pipeline internals at a specific point in the denoising loop. Reading their source code carefully before wrapping is essential.

---

### 7.5 PAES Metric

**What it is:** A Python function. Simple to implement once you have real numbers from the benchmark.

**Steps:**
1. Run baseline and WorldCache through the full benchmark — get real speed, physics, drift numbers
2. Implement the formula, confirm the rankings feel intuitive
3. Add normalization so speed and physics scores are on comparable scales
4. Add PAES to every benchmark run output automatically

**What you're coding:**
```python
# worldoptbench/metrics/paes.py

def compute_paes(speedup, physics_score, drift_rate, horizon_t):
    drift_penalty = 1 + abs(drift_rate) * horizon_t
    return (speedup * physics_score) / drift_penalty
```

**Hardest part:** Justifying the formula weights in the paper — run ablations showing rankings are robust to small changes.

---

### 7.6 AdaCache + Quantization Modules

**What it is:** Same pattern as WorldCache wrapper, applied to two more techniques.

**Steps:**
1. Wrap AdaCache — same process as WorldCache, different hook point in the pipeline
2. Wrap FP8 quantization using TorchAO — apply to model weights before generation, architecture agnostic
3. Add both to the benchmark, run full comparison table

**What you're coding:**
```
worldoptbench/optimizations/
├── adacache.py        ← AdaCache wrapper
└── quantization.py    ← FP8/INT4 via TorchAO
```

---

### 7.7 Speculative Decoding Module

**What it is:** An optimization module based on SDVG (April 2026) that adapts speculative decoding from LLMs to video/world model generation. Complementary to WorldCache — caching reduces the number of denoising steps, speculative decoding parallelizes the verification of those steps.

**How speculative decoding works for diffusion:**
In LLMs, speculative decoding uses a small draft model to propose tokens, then a large model verifies them in parallel — if the draft was right you get multiple tokens for the cost of one large model pass. For diffusion world models the adaptation is:
1. A small draft model (e.g. Cosmos 2B) proposes a denoised frame at step T
2. The large model (Cosmos 7B) verifies it — if close enough, accept and skip the large model's own denoising pass for that step
3. Only run the full large model when the draft diverges significantly

This is particularly powerful for world models because many denoising steps are "easy" — the frame doesn't change much — so the small draft model gets them right most of the time.

**Steps:**
1. Read SDVG paper (`arxiv:2604.17397`) — understand their specific acceptance criterion for video frames
2. Set up the draft model (Cosmos 2B) alongside the target model (Cosmos 7B) — both loaded simultaneously, draft on less VRAM
3. Implement the proposal loop — run draft model for N steps, batch-verify with large model
4. Implement the acceptance criterion — SDVG uses a frame similarity threshold, tune for physics consistency preservation
5. Wrap as an `OptimizationModule` with `supported_architectures = ["diffusion"]`
6. Benchmark against WorldCache alone and combined

**What you're coding:**
```python
# worldoptbench/optimizations/speculative_decoding.py

class SpeculativeDecodingModule(OptimizationModule):
    supported_architectures = ["diffusion"]

    def __init__(
        self,
        draft_model="nvidia/Cosmos-Predict2-2B",   # small proposer
        acceptance_threshold=0.85,                  # frame similarity cutoff
        draft_steps=4,                              # how many steps draft proposes
    ):
        self.draft_model = draft_model
        self.threshold = acceptance_threshold
        self.draft_steps = draft_steps

    def apply(self, pipeline, config):
        # Load draft model alongside target
        # Intercept denoising loop
        # Run draft for N steps, verify with target
        # Accept or reject based on frame similarity
        ...
```

**How it combines with WorldCache:**
```python
stack = OptimizationStack(
    model="nvidia/cosmos-predict-7b",
    modules=[
        "worldcache",            # reduces redundant steps via caching
        "speculative_decoding",  # parallelizes verification of remaining steps
        "drift_correction",      # preserves physics under both
    ]
)
```

**VRAM note:** Running both draft and target model simultaneously needs more VRAM — on a single H100 this means the draft (2B) and target (7B) together need ~50GB. Works on H100 80GB, tight on A100 40GB. With quantization on the draft model it fits comfortably.

**Hardest part:** The acceptance criterion — too strict and you reject most drafts (no speedup), too loose and physics consistency degrades. Needs careful tuning and benchmarking on PAES, not just speed.

**Note on AR models:** Standard LLM-style speculative decoding applies directly to autoregressive world models — add as a separate AR-specific module when AR support is added in future work.

---

### 7.9 Drift Correction Module (Novel Research Piece)

**What it is:** A PyTorch hook that intercepts the denoising loop, detects physics drift onset, and selectively recomputes drifting tokens.

**Steps:**
1. Read WorldCache source thoroughly — understand how curvature scores are computed per token
2. Implement the drift monitor — optical flow divergence between adjacent frames as drift proxy
3. Implement selective recomputation — recompute top-K tokens by curvature score when threshold crossed
4. Blend corrections into cached state
5. Wrap as an `OptimizationModule` with `supported_architectures = ["diffusion"]`
6. Benchmark: confirm Pareto improvement on PAES vs WorldCache alone

**What you're coding:**
```python
# worldoptbench/optimizations/drift_correction.py

class DriftCorrectionModule(OptimizationModule):
    supported_architectures = ["diffusion"]

    def __init__(self, threshold=0.15, top_k_tokens=64):
        self.threshold = threshold
        self.top_k = top_k_tokens

    def apply(self, pipeline, config):
        # Register forward hooks on DiT
        # Monitor optical flow divergence
        # Trigger selective recomputation at threshold
        # Blend back into WorldCache state
        ...
```

**Hardest part:** Hooking into the diffusion denoising loop cleanly. Requires solid PyTorch internals knowledge — `register_forward_hook`, understanding the Diffusers scheduler, manipulating cached tensors mid-generation.

**Fallback:** If it doesn't work in time, the benchmark + PAES + architecture-agnostic framework is already publishable.

---

### 7.10 Distributed Inference + CUDA Memory Management

**What it is:** Multi-GPU support via tensor parallelism and automatic CUDA memory handling. Lets the framework run 7B models at 720p without OOM errors.

**Steps:**
1. Add Accelerate to the `WorldModelServer` init — wrap the loaded model with `accelerate.dispatch_model()` which handles tensor parallelism automatically
2. Write the VRAM checker — on startup, estimate model memory requirements based on parameter count + dtype, compare against available VRAM, suggest config if insufficient
3. Write the dtype auto-selector — detect GPU type (H100 → bfloat16, A100 → float16, RTX → int8) and set accordingly
4. Expose distributed config in `WorldModelServer` constructor
5. Add multi-GPU VRAM tracking to benchmark runner — report per-GPU peak usage

**What you're coding:**
```
worldoptbench/
└── distributed/
    ├── __init__.py
    ├── tensor_parallel.py     ← Accelerate wrapper for tensor parallelism
    ├── memory.py              ← VRAM checker, dtype auto-selector
    └── profiler.py            ← per-GPU memory tracking for benchmarks
```

**Hardest part:** Making sure quantization modules and tensor parallelism don't conflict — some quantization approaches don't play well with weight sharding across GPUs. Test combinations carefully.

**What stays in future work:** Pipeline parallelism and spatial parallelism — more complex, not needed for v1 hardware targets.

---

### 7.11 `worldserve` Serving Layer

**What it is:** FastAPI server wrapping the optimization stack. Do this last — it's a product layer on top of everything above.

**Steps:**
1. Write `WorldModelServer` class — initializes model + stack on startup, holds them in memory
2. Write FastAPI app — single `/generate` POST endpoint, returns frames + PAES profile
3. Add async request queue + batching
4. Write Dockerfile
5. Write CLI entry point — `worldserve start --model cosmos-predict-7b`

**What you're coding:**
```
worldserve/
├── server.py      ← WorldModelServer class
├── app.py         ← FastAPI app
├── batching.py    ← async queue + batcher
├── cli.py         ← CLI
└── Dockerfile
```

**Hardest part:** Async batching with GPU models — handling requests mid-generation without blocking. Reference vLLM's continuous batching implementation.

---

## 8. Compute & Cost Plan

| Task | Hardware | Est. GPU-hrs | Est. Cost |
|------|----------|-------------|-----------|
| Setup + baseline runs | H100 rented | ~20 hrs | ~$50 |
| WorldCache + AdaCache reproduction | H100 rented | ~40 hrs | ~$100 |
| Long-horizon sweep (4s/16s/60s/120s) | H100 rented | ~60 hrs | ~$150 |
| Drift correction dev + ablations | H100 rented | ~120 hrs | ~$300 |
| Final benchmark (all methods, all models) | H100 rented | ~40 hrs | ~$100 |
| **Total** | | **~280 hrs** | **~$700** |

**Free options:**
- University HPC cluster — highest priority, free if you can get access
- Google TPU Research Cloud — free, apply at trc.devsite.google.com
- Google Research Credits — $500 GCP credits
- RunPod / Lambda Labs — ~$1.50–2.00/hr H100 spot

---

## 9. Timeline

```
Month 1 — Foundation
  Week 1–2:  WorldModelInterface + Cosmos-Predict running, baseline generation working
  Week 3–4:  Benchmark harness v1 (speed + visual metrics), first long-horizon runs

Month 2 — Empirical Results
  Week 5–6:  Physics metrics integrated (PAI-Bench, WorldRoamBench), full benchmark sweep
  Week 7–8:  WorldCache wrapped + benchmarked, PAES metric defined and validated

Month 3 — Novel Method
  Week 9–10: AdaCache + quantization modules, full comparison table
  Week 11–12: Drift correction implementation + testing

Month 4 — Polish
  Week 13–14: Ablations, architecture-agnostic validation (Open-Sora)
  Week 15–16: Paper draft, worldserve v0.1, library cleanup

Month 5 — Ship
  Workshop / conference submission
  GitHub release (WorldOptBench + worldserve)
```

**Milestones:**
- [ ] Cosmos-Predict generating frames — Week 2
- [ ] Benchmark harness running speed + visual metrics — Week 4
- [ ] Physics metrics integrated — Week 6
- [ ] WorldCache reproduced + wrapped — Week 7
- [ ] PAES metric validated — Week 8
- [ ] Full comparison table (baseline, WorldCache, AdaCache, quant) — Week 10
- [ ] Drift correction prototype working — Week 12
- [ ] Paper draft — Week 15
- [ ] Public GitHub release — Week 16

---

## 10. Startup Path

**The product:** Managed inference API for world models with physics-aware optimization — architecture agnostic.

**Customers:**
- Robotics companies generating synthetic training data
- AV teams running edge case simulation
- Game studios doing procedural generation

**Why now:** NVIDIA Cosmos has 2M downloads. Every team hits the same wall. We're the first architecture-agnostic framework with physics as a first-class metric.

**Path:**
1. Open-source framework → GitHub stars → community credibility
2. Hosted API endpoint → first paying customers
3. YC (explicitly funding physical AI startups in 2026 batch)
4. Enterprise contracts with robotics / AV companies

---

## 11. Reading List

**Must-read:**
- WorldCache v1 `arxiv:2603.06331`
- WorldCache v2 `arxiv:2603.22286`
- OpenWorldLib `arxiv:2604.04707`
- WorldRoamBench `arxiv:2606.31672`
- "Towards Interactive Video World Modeling" `arxiv:2606.01164`
- Cosmos technical report `arxiv:2501.03575`

**Background:**
- AdaCache `arxiv:2411.02397`
- AccVideo `arxiv:2503.19462`
- SDVG (speculative decoding for video) `arxiv:2604.17397`
- Fei-Fei Li taxonomy paper (Jun 3 2026)
- V-JEPA 2 technical report (Meta, 2025)

**Videos:**
- Inference: https://www.youtube.com/watch?v=B18zBnjZKmc
- Optimizing inference: https://www.youtube.com/watch?v=hMs8VNRy5Ys
- Quantization: https://www.youtube.com/watch?v=qoQJq5UwV1c
- Inference engines: https://www.youtube.com/watch?v=uqUZ_H_m2Yg
- World models: https://www.youtube.com/watch?v=MqjvfJTCuqw
