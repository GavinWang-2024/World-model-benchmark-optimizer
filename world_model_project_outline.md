# WorldOptBench — Project Outline

> **One-liner:** A general framework for optimizing and benchmarking world model inference — treating speed, visual quality, and physics consistency as a joint objective for the first time.

---

## 1. The Problem

World model inference is **8–32x more expensive** than LLM inference. A wave of optimization papers has emerged in 2026:

| Paper | Technique | Speedup | Physics Eval? |
|-------|-----------|---------|---------------|
| WorldCache (Mar 2026) | Heterogeneous token caching | 2.3x | ❌ |
| AdaCache | Adaptive step caching | 4.7x | ❌ |
| AccVideo | Distillation | 8.5x | ❌ |
| OpenWorldLib (Apr 2026) | Unified inference codebase | — | ❌ |

**Every single one optimizes for speed + visual quality only.**

For the actual customers — robotics teams, AV simulation, game studios — **physics consistency and long-horizon stability matter as much as speed.** Nobody has built a framework that treats all three as a joint objective, and nobody has a standard way to benchmark across them.

**Confirmed open problems from the literature:**
- "The field still lacks unified, interpretable, and closed-loop-relevant evaluation protocols for long-horizon stability" — *Latent World Models for Automated Driving, 2026*
- "Maintaining coherent world evolution over extremely long horizons remains one of the most critical open problems" — *Towards Interactive Video World Modeling, 2026*
- WorldRoamBench (Jun 2026) evaluated 10+ models — none reliably satisfies all dimensions

---

## 2. What We're Building

**WorldOptBench** — a general framework with two components:

```
WorldOptBench
├── Benchmark Suite        ← standardized eval across speed + visual + physics
└── Optimization Stack     ← plug-in modules, jointly tunable
    └── Serving Layer      ← production inference server (worldserve)
```

The thesis: **if you can't measure all three axes jointly, you can't optimize them jointly.** We build the measurement standard first, then the optimization stack on top of it.

---

## 3. Core Features (v1 — What We Actually Ship)

These are the non-negotiable deliverables for a first version. Everything else is future work.

### 3.1 Unified Benchmark Suite

A single eval harness that runs any optimization technique against any open world model and outputs a standardized profile across three axes:

**Axis 1 — Speed**
- Wall-clock latency (seconds per generated second of video)
- Tokens/frames per second
- GPU memory peak usage
- Time to first frame

**Axis 2 — Visual Quality**
- FVD (Fréchet Video Distance) — standard
- PSNR / SSIM per frame
- Temporal consistency (frame-to-frame delta)

**Axis 3 — Physics Consistency** ← the new one
- PAI-Bench score (physical plausibility + controllability)
- WorldRoamBench physics subscore
- Long-horizon drift curve (physics score vs. rollout length at 4s / 16s / 60s / 120s)

**Output for every run:**
```
Model:        Cosmos-Predict 7B
Optimization: WorldCache
Hardware:     H100

Speed:        2.3x speedup | 18.4s/s generation | 42GB VRAM peak
Visual:       FVD 84.2 | PSNR 31.1 | Temporal consistency 0.94
Physics:      PAI-Bench 0.71 | Drift onset: 28s | Drift rate: -0.008/s

PAES Score:   1.84   ← our combined metric (see below)
```

**Supported models (v1):**
- Cosmos-Predict 7B (primary)
- Cosmos-Predict 2.5-2B (lightweight experiments)
- Open-Sora (secondary)

**Supported optimizations (v1 — existing techniques as inputs):**
- Baseline (no optimization)
- WorldCache
- AdaCache
- FP8 quantization
- Combinations of the above

---

### 3.2 Physics-Aware Efficiency Score (PAES)

A single combined metric so optimization papers have one number to report and compare against:

```
PAES = Speedup × Physics_Score × (1 / Drift_Rate_at_T)
```

Where:
- `Speedup` = baseline latency / optimized latency
- `Physics_Score` = PAI-Bench or WorldRoamBench physics subscore (0–1)
- `Drift_Rate_at_T` = slope of physics degradation curve at rollout horizon T

**Why this matters:** A method that gets 4x speedup with 50% physics degradation should score lower than one that gets 2x speedup with 5% degradation. Current papers can't express this. PAES can.

This alone is a publishable contribution as a metric/benchmark paper.

---

### 3.3 Optimization Stack (Plug-in Architecture)

A modular system where optimization techniques are composable components:

```python
from worldoptbench import OptimizationStack

stack = OptimizationStack(
    model="nvidia/cosmos-predict-7b",
    modules=[
        "worldcache",        # existing — plug in as-is
        "adacache",          # existing — plug in as-is
        "fp8_quantization",  # existing — plug in as-is
        "drift_correction",  # ours — the novel piece
    ],
    constraints={
        "min_physics_score": 0.85,   # don't sacrifice physics below this
        "target_speedup": 2.0,       # aim for this speedup
        "max_vram_gb": 80,           # H100 SXM
    }
)
```

Modules are composable and the stack auto-selects the combination that satisfies the constraints. v1 ships with existing techniques as modules; our novel method (drift correction) is one more module added on top.

---

### 3.4 Serving Layer (`worldserve`)

Production inference server wrapping the optimization stack. The gap vs. OpenWorldLib: OpenWorldLib is a research codebase for *calling* models. `worldserve` is a server you actually deploy.

```python
pip install worldserve

from worldserve import WorldModelServer

server = WorldModelServer(
    model="nvidia/cosmos-predict-7b",
    stack=["worldcache", "drift_correction"],
    hardware="H100",
    physics_budget=0.90,   # minimum PAES physics component
    speed_budget=2.0,      # target speedup
)

# HTTP endpoint — call from anywhere
server.serve(port=8000)

# Or direct Python API
frames = server.generate(
    prompt="robot arm picking up red cube on warehouse floor",
    horizon=60,
    action_sequence=[...],
)
```

**v1 features:**
- Works on single H100 (480p) or RTX 4090 with quantization (360p)
- FastAPI endpoint, async batching
- Returns PAES profile alongside every generation
- Docker container for easy deployment

**The analogy:** vLLM is to LLMs what `worldserve` is to world models.

---

## 4. The Novel Research Piece

The framework *needs* something new to anchor a paper — we can't just wrap existing work. The novel contribution is:

### Inference-Time Physics Drift Correction

**The problem:** WorldCache showed that "a small set of hard tokens drives error growth." Current methods either recompute everything (slow) or skip uniformly (loses physics). Neither is optimal.

**Our mechanism (training-free):**
1. **Monitor** a lightweight physics consistency signal during rollout — optical flow divergence or curvature score (WorldCache already computes this, we repurpose it)
2. **Detect** when the signal crosses a threshold indicating drift onset
3. **Correct** by selectively recomputing only the chaotic tokens, not the full frame
4. **Blend** the correction into the cached state

This runs *on top of* WorldCache as a module. No retraining. No new model weights.

**Target:** Match WorldCache's 2.3x speedup while recovering 60–80% of the physics consistency gap vs. unoptimized baseline. A clear Pareto improvement on the PAES metric.

---

## 5. Future Work (Post-v1)

Things we explicitly do not build in v1 but could add:

- **More models:** Genie 3 (if API access), WAN2.1, custom fine-tunes
- **More optimization modules:** Speculative decoding for AR video (SDVG), distillation pipelines, INT4 quantization
- **Autoscaling:** Multi-GPU scheduling, disaggregated prefill/decode for world models
- **Domain-specific physics metrics:** Robotics-specific (contact dynamics, grasp stability), AV-specific (trajectory plausibility, agent behavior)
- **LLM optimization sidebar:** Dynamic depth routing, attention residuals — lower priority, track separately
- **Managed API product:** If the library gets traction, productize as a hosted endpoint

---

## 6. Related Work & How We Differ

| Project | What It Does | Gap |
|---------|-------------|-----|
| WorldCache | Caching optimization, 2.3x speedup | No physics eval, no framework |
| AdaCache | Step-skipping, 4.7x speedup | Same gap |
| OpenWorldLib | Unified inference codebase | No optimization, no physics eval |
| WorldRoamBench | Long-horizon benchmark | Benchmark only, no optimization |
| PAI-Bench | Physics plausibility eval | No speed optimization component |
| **WorldOptBench (ours)** | **Framework: joint optimization + benchmarking across all three axes** | — |

---

## 7. Supported Models (v1)

| Model | Open Weight | Min Hardware | Notes |
|-------|-------------|-------------|-------|
| Cosmos-Predict 7B | ✅ | H100 (480p) | Primary target |
| Cosmos-Predict 2.5-2B | ✅ | RTX 4090 | Lightweight experiments |
| Open-Sora | ✅ | A100 | Secondary target |

---

## 8. Compute & Cost Plan

| Task | Hardware | Est. GPU-hrs | Est. Cost |
|------|----------|-------------|-----------|
| Environment setup + baseline runs | H100 rented | ~20 hrs | ~$50 |
| WorldCache + AdaCache reproduction | H100 rented | ~40 hrs | ~$100 |
| Long-horizon sweep (4s/16s/60s/120s) | H100 rented | ~60 hrs | ~$150 |
| Drift correction dev + ablations | H100 rented | ~120 hrs | ~$300 |
| Final benchmark runs (all methods) | H100 rented | ~40 hrs | ~$100 |
| **Total** | | **~280 hrs** | **~$700** |

**Free options:**
- University HPC cluster — highest priority, free if you can get access
- Google TPU Research Cloud — free, apply at trc.devsite.google.com
- Google Research Credits — $500 GCP credits for researchers
- RunPod/Lambda Labs — ~$1.50–2.00/hr H100 spot, much cheaper than AWS

---

## 9. Timeline

```
Month 1 — Foundation
  Week 1–2:  Cosmos-Predict setup, reproduce WorldCache baseline
  Week 3–4:  Build benchmark harness v1, first long-horizon runs

Month 2 — Empirical Results
  Week 5–6:  Full benchmark sweep (all methods, all horizons)
  Week 7–8:  Define + validate PAES metric, draft benchmark results

Month 3 — Novel Method
  Week 9–10: Design drift correction mechanism
  Week 11–12: Implement + test drift correction module

Month 4 — Polish
  Week 13–14: Ablations, full comparison table
  Week 15–16: Paper draft, worldserve v0.1 cleanup

Month 5 — Ship
  Workshop/conference submission
  GitHub release (WorldOptBench + worldserve)
```

**Milestones:**
- [ ] Cosmos-Predict running — Week 2
- [ ] WorldCache reproduced — Week 3
- [ ] Benchmark harness running all three axes — Week 4
- [ ] Long-horizon drift curves complete — Week 6
- [ ] PAES metric validated — Week 8
- [ ] Drift correction prototype — Week 12
- [ ] Full paper draft — Week 15
- [ ] Public GitHub release — Week 16

---

## 10. Startup Path

**The product:** Managed inference API for world models with physics-aware optimization.

**Customers:**
- Robotics companies generating synthetic training data
- AV teams running edge case simulation
- Game studios doing procedural generation

**Why now:** NVIDIA Cosmos has 2M downloads. Every team using it hits the same wall — too slow, no physics guarantees, no standard way to measure tradeoffs. We're the first to give them that.

**Path:**
1. Open-source framework → GitHub stars → community credibility
2. Hosted API endpoint → first paying customers
3. YC application (explicitly funding physical AI startups in 2026)
4. Enterprise contracts with robotics/AV companies

---

## 11. Reading List

**Must-read:**
- WorldCache v1 `arxiv:2603.06331` — caching prior work
- WorldCache v2 `arxiv:2603.22286` — second approach
- OpenWorldLib `arxiv:2604.04707` — existing framework
- WorldRoamBench `arxiv:2606.31672` — benchmark we build on
- "Towards Interactive Video World Modeling" `arxiv:2606.01164` — open problems
- Cosmos technical report `arxiv:2501.03575` — target model

**Background:**
- AdaCache `arxiv:2411.02397`
- AccVideo `arxiv:2503.19462`
- SDVG `arxiv:2604.17397` — speculative decoding for video
- Fei-Fei Li taxonomy paper (Jun 3 2026)

**Videos:**
- Inference: https://www.youtube.com/watch?v=B18zBnjZKmc
- Optimizing inference: https://www.youtube.com/watch?v=hMs8VNRy5Ys
- Quantization: https://www.youtube.com/watch?v=qoQJq5UwV1c
- Inference engines: https://www.youtube.com/watch?v=uqUZ_H_m2Yg
- World models: https://www.youtube.com/watch?v=MqjvfJTCuqw

---

## 12. General Technologies

**Core language:** Python 3.11+

**ML / Model layer:**
- PyTorch — tensor operations, hooking into diffusion model internals for drift correction
- HuggingFace Diffusers — base library Cosmos-Predict and Open-Sora are built on, how we load and run models
- HuggingFace Hub — model weight downloads
- xFormers / Flash Attention 2 — memory-efficient attention, already used by target models

**Optimization modules:**
- WorldCache / AdaCache — pulled directly from their open-source repos, wrapped as plug-in modules
- BitsAndBytes / TorchAO — FP8 and INT4 quantization
- CUDA / Triton — only if we write custom kernels (not in v1, future work)

**Benchmarking:**
- PyTorch profiler + NVIDIA `nvitop` / `nvidia-smi` — GPU memory and utilization tracking
- `torchmetrics` — FVD, PSNR, SSIM computation
- PAI-Bench + WorldRoamBench — pulled from their public repos for physics scoring
- NumPy / pandas — storing and processing benchmark results
- Matplotlib / seaborn — Pareto frontier plots, drift curves for the paper

**Serving layer:**
- FastAPI — HTTP server for `worldserve`
- Uvicorn — ASGI server
- Pydantic — request/response validation
- Docker — containerized deployment

**Infrastructure:**
- RunPod / Lambda Labs — rented H100s for experiments
- GitHub Actions — CI for the open-source library
- pytest — test suite

---

## 13. Implementation Approach — Feature by Feature

### 13.1 Benchmark Harness

**What it is:** A script (and eventually a CLI) that takes a model + optimization config, runs a standardized set of rollouts, and outputs a structured results file.

**Steps:**
1. Define a standard rollout set — fixed set of prompts and action sequences at 4s / 16s / 60s / 120s horizons so all methods are compared on identical inputs
2. Write the base runner — loads Cosmos-Predict via HuggingFace Diffusers, runs generation, captures wall-clock time and VRAM peak using PyTorch profiler
3. Plug in visual quality metrics — compute FVD, PSNR, SSIM per rollout using `torchmetrics`
4. Plug in physics metrics — run PAI-Bench and WorldRoamBench scorers on the generated frames
5. Write results to a structured JSON file per run
6. Write a comparison script that takes multiple result JSONs and produces the Pareto plots

**What you're coding:**
```
worldoptbench/
├── runner.py          ← orchestrates a full benchmark run
├── metrics/
│   ├── speed.py       ← latency, throughput, VRAM
│   ├── visual.py      ← FVD, PSNR, SSIM
│   └── physics.py     ← PAI-Bench, WorldRoamBench wrappers
├── prompts/
│   └── standard_set.json  ← the fixed rollout set
└── compare.py         ← takes N result JSONs, outputs plots
```

**Hardest part:** Getting PAI-Bench and WorldRoamBench to run cleanly — both are research codebases with rough edges. Expect setup friction.

---

### 13.2 PAES Metric

**What it is:** A Python function. Genuinely simple to implement once you have the benchmark harness outputting the three component scores.

**Steps:**
1. Confirm the formula with experiments — run baseline Cosmos-Predict and WorldCache, make sure the scores feel right and rank methods intuitively
2. Handle the horizon parameter T — PAES should be reported at a specific rollout length since drift is horizon-dependent
3. Write a normalizer — raw PAI-Bench scores and speedup numbers are on different scales, normalize before combining
4. Add PAES output to every benchmark run automatically

**What you're coding:**
```python
# worldoptbench/metrics/paes.py

def compute_paes(speedup: float, physics_score: float, drift_rate: float, horizon_t: float) -> float:
    """
    speedup: baseline_latency / optimized_latency
    physics_score: PAI-Bench or WorldRoamBench physics subscore (0-1)
    drift_rate: slope of physics degradation curve at horizon T (per second)
    horizon_t: rollout length in seconds
    """
    drift_penalty = 1 + abs(drift_rate) * horizon_t
    return (speedup * physics_score) / drift_penalty
```

**Hardest part:** Justifying the formula in the paper. Reviewers will push back on the specific weighting — you need ablations showing PAES rankings are robust to small formula changes.

---

### 13.3 Optimization Stack (Plug-in Architecture)

**What it is:** A base class + registry pattern so optimization techniques are composable modules that can be stacked and configured.

**Steps:**
1. Define a base `OptimizationModule` abstract class with a standard interface — `apply(pipeline, config) -> pipeline`
2. Wrap WorldCache as a module — clone their repo, import their core caching logic, adapt it to the interface
3. Wrap AdaCache as a module — same process
4. Wrap FP8 quantization as a module — using TorchAO or BitsAndBytes
5. Write the `OptimizationStack` class that chains modules together and handles conflicts (e.g. two caching methods that can't both run)
6. Write the constraint solver — given `min_physics_score` and `target_speedup`, auto-select the best module combination based on benchmark results

**What you're coding:**
```
worldoptbench/
├── optimizations/
│   ├── base.py            ← OptimizationModule abstract class
│   ├── worldcache.py      ← WorldCache wrapper
│   ├── adacache.py        ← AdaCache wrapper
│   ├── quantization.py    ← FP8/INT4 wrapper
│   └── drift_correction.py ← our novel module (see 13.4)
└── stack.py               ← OptimizationStack, constraint solver
```

**Hardest part:** WorldCache and AdaCache hook into the diffusion pipeline at different points. Wrapping them cleanly without breaking each other requires reading their source code carefully and understanding HuggingFace Diffusers' pipeline internals.

---

### 13.4 Drift Correction Module (Novel Research Piece)

**What it is:** A PyTorch hook that intercepts the denoising loop mid-rollout, detects physics drift onset, and selectively recomputes drifting tokens.

**Steps:**
1. Read WorldCache source code thoroughly — understand exactly how they compute the curvature score per token and how the cache is structured
2. Implement the drift monitor — a lightweight signal computed at each generation step using optical flow divergence between adjacent frames. When the signal crosses a threshold, flag drift onset
3. Implement selective recomputation — instead of recomputing the full frame (slow) or skipping entirely (loses physics), recompute only the top-K tokens by WorldCache curvature score
4. Implement the cache blend — write the corrected tokens back into the WorldCache state cleanly so subsequent cached steps build on the corrected state
5. Expose it as an `OptimizationModule` so it slots into the stack alongside WorldCache

**What you're coding:**
```python
# worldoptbench/optimizations/drift_correction.py

class DriftCorrectionModule(OptimizationModule):
    def __init__(self, threshold: float = 0.15, top_k_tokens: int = 64):
        self.threshold = threshold   # optical flow divergence threshold
        self.top_k = top_k_tokens    # how many tokens to selectively recompute

    def apply(self, pipeline, config):
        # Register forward hooks on the diffusion UNet/DiT
        # Monitor optical flow divergence per step
        # Trigger selective recomputation when threshold crossed
        # Blend corrections into cached state
        ...
```

**Hardest part:** This is the hardest piece in the whole project. Hooking into a diffusion model's denoising loop without breaking the generation requires solid PyTorch internals knowledge. Expect this to take the most time and debugging.

**Fallback if it doesn't work:** The benchmark + PAES metric alone is publishable. Drift correction is a bonus, not a requirement for a workshop paper.

---

### 13.5 `worldserve` Serving Layer

**What it is:** A FastAPI server that wraps the optimization stack and exposes it as an HTTP API you can call from anywhere.

**Steps:**
1. Write the `WorldModelServer` class — takes model name, stack config, hardware config, initializes everything on startup
2. Write the FastAPI app — single `/generate` POST endpoint that accepts prompt + horizon + action sequence, returns frames + PAES profile
3. Add async batching — queue incoming requests and batch them for GPU efficiency
4. Write the Docker container — `Dockerfile` that handles CUDA dependencies cleanly
5. Write a simple CLI — `worldserve start --model cosmos-predict-7b --hardware H100`

**What you're coding:**
```
worldserve/
├── server.py          ← WorldModelServer class
├── app.py             ← FastAPI app, /generate endpoint
├── batching.py        ← async request queue + batcher
├── cli.py             ← CLI entry point
└── Dockerfile
```

**Hardest part:** Async batching with GPU models is tricky — you need to handle requests that come in while the GPU is mid-generation, queue them, and flush the batch without introducing too much latency. Look at vLLM's continuous batching implementation for reference.

---

### Build Order (What to Code First)

Do these in order — each one unblocks the next:

1. **Benchmark runner (13.1)** — get Cosmos-Predict running and measurable first, everything else depends on this
2. **Visual + speed metrics (13.1)** — straightforward, do alongside the runner
3. **Physics metrics (13.1)** — expect setup friction, do early so you hit problems early
4. **WorldCache wrapper (13.3)** — first optimization module, validates the plug-in architecture
5. **PAES metric (13.2)** — implement once you have real numbers from above
6. **AdaCache + quantization wrappers (13.3)** — same pattern as WorldCache, faster to implement
7. **Drift correction (13.4)** — the hard part, do after everything else is working
8. **`worldserve` server (13.5)** — do last, it's a product layer on top of everything above
