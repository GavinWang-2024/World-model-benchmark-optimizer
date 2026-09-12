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

> **⚠️ Needs re-scoping (found 2026-09-12):** WorldCache's own mechanism already does curvature-guided drift detection + selective recompute of "bottleneck tokens" — the core loop described above. WorldForge (optical-flow trajectory drift correction) and Polestar (drift-aware cache calibration) cover overlapping ground too. Building this exactly as written risks re-deriving an existing technique. See §5 for the three overlapping papers. Candidate pivots, not yet decided:
> 1. **Make it composable across optimization backends** — a drift-correction layer that plugs onto *any* caching method (WorldCache, AdaCache, or future ones) rather than one paper's fixed technique baked in. The novelty shifts from "a drift correction algorithm" to "drift correction as a portable safety net any optimization module can sit under" — genuinely framework-level, not a single fixed method.
> 2. **Target quantization-induced drift, not caching-induced drift** — the quantization empirical study (`2602.02110`, §5) shows quantization degrades temporal/physics metrics first, but only measures it, doesn't correct it. Nobody's built an active correction mechanism for *quantization*-caused physics drift specifically (as opposed to caching-caused drift, which is now crowded).
> 3. **Extend drift correction to autoregressive models** — every existing drift-correction paper found is diffusion-specific. An AR-model equivalent (correcting drift introduced by KV-cache-based skipping) is still open.
> Recommend deciding this before Phase 7 (build_plan.md) — doesn't block Phases 0-6.

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
- **Autoscaling:** Multi-GPU scheduling, disaggregated prefill/decode
- **Domain-specific physics metrics:** Robotics (contact dynamics, grasp stability), AV (trajectory plausibility)
- **Managed API product:** If the library gets traction, productize as a hosted endpoint
- **LLM optimization sidebar:** Dynamic depth routing, attention residuals — lower priority

---

## 5. Related Work & How We Differ

*(Verified against current literature 2026-09-12 — see note below the table.)*

| Project | What It Does | Gap |
|---------|-------------|-----|
| WorldCache (`2603.06331`, `2603.22286`) | Diffusion caching via curvature-guided token prediction; **built-in drift detection + selective recompute of "bottleneck tokens"** — 2.3–3.7x speedup | Diffusion only, no physics eval. ⚠️ Its own drift-detection mechanism overlaps with our planned §3.5 — see caveat below |
| AdaCache | Step-skipping, 4.7x speedup | Diffusion only, no physics eval |
| WorldForge | Optical-flow-based motion/appearance decoupling + "Dual-Path Self-Corrective Guidance" to correct trajectory drift | Diffusion only, single fixed technique, no benchmark suite, no physics score, not composable with other optimizations |
| Polestar | Drift-aware cache calibration + token commitment (diffusion **LLMs**, not video) | Different modality; same core idea (drift-aware caching) already applied elsewhere |
| OpenWorldLib (`2604.04707`) | Unified inference codebase across video gen / 3D gen / VLA tasks; **does have a real benchmark pipeline** (FVD/FID/SSIM/LPIPS) | Visual-quality metrics only — no physics axis, no speed axis reported, no optimization modules |
| WorldLens (`2512.10958`) | Driving-domain benchmark across 5 dims incl. physics + geometry + control, human-annotated (WorldLens-26K) | Evaluation-only, driving-specific, **no speed/inference-cost axis at all** |
| WorldBench | Isolates physics concepts for diagnostic eval | Evaluation-only, no speed, no optimization |
| DISK | Dynamic inference skipping preserving physics/stability, for AV world models | Single technique, domain-specific (driving), not a general framework |
| Inferix | Inference engine spanning AR + diffusion via semi-autoregressive decoding | Serving/decoding engine, not a benchmark; no physics axis |
| WorldRoamBench | Long-horizon stability benchmark | Benchmark only, no optimization |
| PAI-Bench (`2512.01989`) | Physics plausibility eval, 2,808 real-world cases across AV/robotics/industry/ego-centric | No speed component, no cross-architecture comparison |
| An Empirical Study of World Model Quantization (`2602.02110`) | Shows quantization error propagates differently depending on where it's introduced (representation vs. predictor), and shows up first in temporal/motion metrics before visual ones | Empirical study only — measures the problem, doesn't correct for it, no unified framework |
| **WorldOptBench (ours)** | **Architecture-agnostic framework: joint optimization + benchmarking across all three axes, single PAES metric** | — |

**Caveat (2026-09-12):** No single existing project combines all three — architecture-agnostic + joint speed/visual/physics scoring + a pluggable optimization stack — so the framework-level thesis still holds. But this space is moving fast, and OpenWorldLib (missing physics) and WorldLens (missing speed) are each one axis away from closing part of the gap. The originally-planned drift-correction module (§3.5) needs re-scoping — see the note there.

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
- CUDA / Triton — only if custom kernels needed (not in v1)

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
7. Drift correction module  ← hardest
8. worldserve server        ← last
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
│   ├── base.py           ← OptimizationModule abstract class
│   └── worldcache.py     ← WorldCache wrapper
└── stack.py              ← OptimizationStack + constraint solver
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

### 7.7 Drift Correction Module (Novel Research Piece)

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

### 7.8 `worldserve` Serving Layer

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

**Added 2026-09-12 (from competitive-landscape check — see §5 and §12):**
- WorldForge — optical-flow trajectory drift correction, `arxiv:2509.15130`
- Polestar — drift-aware cache calibration for diffusion LLMs, `arxiv:2607.14107`
- WorldLens — driving-domain 5-axis benchmark incl. physics, `arxiv:2512.10958`
- PAI-Bench full paper — `arxiv:2512.01989`
- An Empirical Study of World Model Quantization — `arxiv:2602.02110` (relevant to Phase 6 quantization module and the §3.5 pivot options)
- DISK (Dynamic Inference SKipping, AV world models)
- Inferix — semi-autoregressive inference engine spanning AR + diffusion

**Videos:**
- Inference: https://www.youtube.com/watch?v=B18zBnjZKmc
- Optimizing inference: https://www.youtube.com/watch?v=hMs8VNRy5Ys
- Quantization: https://www.youtube.com/watch?v=qoQJq5UwV1c
- Inference engines: https://www.youtube.com/watch?v=uqUZ_H_m2Yg
- World models: https://www.youtube.com/watch?v=MqjvfJTCuqw

---

## 12. Open Research Directions (Not Yet Decided)

Everything here is a live option, not a commitment — captured so nothing from the 2026-09-12 competitive-landscape review gets lost before we're actually at the decision point (Phase 7 in `build_plan.md`, after Phases 0-6 are stable). Revisit this section then, pick one, and prune the rest.

**A. Drift correction pivot (see full caveat in §3.5).** Original plan overlaps with WorldCache/WorldForge/Polestar. Three candidate replacements, not mutually exclusive to consider together:
  1. Portable drift correction — a correction layer that sits on top of *any* caching backend (WorldCache, AdaCache, future ones), not one fixed algorithm. Framework-level novelty; harder to execute convincingly in the time available.
  2. Quantization-induced drift correction — actively correct the temporal/physics degradation that `2602.02110` shows quantization causes, rather than just measuring it (which is all that paper does). Reuses the Phase 6 quantization module directly. Currently the front-runner — cleanest gap, least crowded.
  3. Autoregressive-model drift correction — every existing drift-correction paper found is diffusion-only. Blocked until an AR world model is actually in v1 (Genie's API is closed; would need a different AR model or to wait).

**B. Cross-architecture "first" claim.** JEPA is being benchmarked against other approaches, but only in embodied/RL-probing setups (CALVIN, MetaWorld) — not as a generative world model scored on the same speed/visual/physics harness as diffusion and AR models. Getting even a rough V-JEPA 2 wrapper (already listed as future work in §4) onto the WorldOptBench benchmark would let us claim to be first to put diffusion, AR, and JEPA on one joint scale for generation. Worth flagging explicitly as a paper-strength goal, not just a "future model implementation" checkbox — the value is in the comparison being apples-to-apples, not just in supporting a third architecture.

**C. The framework claim itself remains the primary defensible position** (per §5 caveat) regardless of what happens with A and B — no existing project combines architecture-agnostic + joint 3-axis scoring + pluggable optimization. Keep this as the fallback thesis if A and B both stall.

**D. Functional-utility preservation — a possible 4th axis (found 2026-09-12, strongest new candidate).** A separate research line (WorldArena `2602.08971`, WorldEval, dWorldEval, WMBench) already checks whether a world model is useful as a policy evaluator / synthetic-data engine — i.e. whether its scores correlate with real downstream robot task success. Nobody checks whether *optimizing* a world model (caching, quantization) preserves that usefulness, as opposed to just preserving FVD/PSNR/physics scores while silently breaking what a robotics customer actually cares about. Concretely: take a lightweight policy-ranking task (borrow WorldEval/dWorldEval's or RoboTwin's setup) and check whether optimized-vs-baseline world models still rank policies the same way. This doesn't compete with WorldCache/WorldForge/Polestar's territory the way drift correction does — it's a genuinely open intersection. Candidate 4th axis alongside speed/visual/physics, or a standalone add-on metric.

**E. Cost-aware Pareto + hardware recommender (found 2026-09-12).** LLM inference has mature $/token Pareto frontiers (OpenRouter etc.); nothing equivalent exists for world models. Cheap to build on top of what Phase 2 already produces — wrap benchmark throughput numbers with known RunPod/Lambda $/hr rates to answer "given my physics/speed budget, which model+optimization+hardware combo is cheapest." Directly matches how the target customers (§10: robotics/AV/game studios) actually think — in dollars, not relative speedup.

**F. Public leaderboard (found 2026-09-12).** No public world-model-optimization leaderboard exists (InferenceBench is LLM-serving-specific, not world models). A living WorldOptBench leaderboard would be a first, and it's the literal mechanism §10's startup path already assumes ("open-source → GitHub stars → community credibility") but doesn't yet name as a concrete deliverable.

**G. Interface completeness gap (not a new idea, a real hole):** `WorldModelInterface.generate()` (§3.1) only accepts `prompt` + `actions`. Cosmos and most WFMs also support Image2World/Video2World conditioning — starting from a real camera frame, which is how most actual robotics/AV use cases work, not from a pure text description. This should probably be fixed at the interface-definition stage (Phase 1) rather than deferred, since every downstream model wrapper and the benchmark runner build on top of this signature.
