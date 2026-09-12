# WorldOptBench — Build Plan

> Companion to `world_model_project_outline.md` (the spec/pitch doc). That doc says *what* everything is. This doc says *what order to actually build it in* and *what "done" looks like at each step*, so you always know what to open your editor and work on next.

**Guiding principle:** get one model generating frames end-to-end as fast as possible (even if ugly), wrap it in measurement before you touch any optimization, then layer optimizations from safest/most-proven → hardest/most-novel, and build the server last because it depends on everything above being stable. Don't build worldserve or drift correction early — there's nothing to serve or correct yet.

---

## Phase 0 — Setup (before writing any framework code)

**Goal:** you can run *something* on a GPU and see output.

- [ ] Accept the NVIDIA Open Model License on the Cosmos-Predict HF page (self-serve click-through + contact info, not a manual review queue) and generate an HF read token — do this first anyway since nothing downloads without it
- [ ] Pick compute: check university HPC access first (free), then Google TRC/GCP credits, fall back to RunPod/Lambda H100 spot (~$1.50–2/hr)
- [ ] Scaffold the repo: `pyproject.toml`, `worldoptbench/` package skeleton, `pytest` wired up, GitHub Actions CI stub
- [ ] Get Cosmos-Predict's own example script running unmodified on your rented/HPC hardware — confirms CUDA/driver/dependency setup before you add any of your own code on top

**Done when:** you've generated at least one video from Cosmos-Predict using their stock code, on your actual hardware.

**Risk:** CUDA/dependency friction. Budget a full day for this and don't let it bleed into Phase 1.

---

## Phase 1 — `WorldModelInterface` + Cosmos-Predict wrapper

**Goal:** your own abstraction exists and one real model implements it.

- [x] `worldoptbench/models/base.py` — the `WorldModelInterface` abstract class (`generate()`, `get_info()`), with `prompt`/`init_frame`/`init_video`/`actions` conditioning (item G resolved: built in from the start)
- [x] `worldoptbench/models/cosmos_predict.py` — `CosmosPredict2B`/`CosmosPredict7B` wrapping the native `cosmos_predict2` package's `Text2World`/`Video2World` pipelines — **written against NVIDIA's documented API but UNTESTED** (no GPU/package access yet); the `Text2WorldPipeline` class name specifically is inferred by analogy to the confirmed `Video2WorldPipeline` pattern, not confirmed directly — see module docstring
- [x] Contract tests (`tests/test_base.py`) — exercise the interface with a fake implementation, no torch/GPU needed, run in CI now
- [ ] Sanity check: call `.generate()` through *your* interface (not their example script) and get frames out — **blocked on Phase 0** (HF token + compute + confirming the real `cosmos_predict2` package layout)

**⚠️ Also found while writing this:** NVIDIA's own docs show Cosmos-Predict2 model sizes as **2B and 14B**, not 7B — the outline's "Cosmos-Predict 7B" (§3.1) may be the wrong size name. Confirm the actual available checkpoint during Phase 0 and fix `CosmosPredict7B`'s `_SIZE`/class name (and the outline's model table) accordingly.

**Done when:** `CosmosPredict2B().generate(prompt=..., horizon=...)` returns real frames, called generically through the interface, on actual hardware.

**Skip for now:** Open-Sora, Genie 3, V-JEPA 2 — one model is enough to build everything else against. Add the second model only once the benchmark runner (Phase 2) exists, as a test that your abstraction is actually architecture-agnostic and not secretly Cosmos-shaped.

---

## Phase 2 — Benchmark runner (speed + visual only)

**Goal:** a script that takes any `WorldModelInterface` and produces a results JSON. No physics yet — that's Phase 3.

- [ ] `prompts/standard_set.json` — fixed prompts + action sequences at 4s/16s/60s/120s horizons
- [ ] `runner.py` — calls `.generate()`, wraps with PyTorch profiler for wall-clock + VRAM
- [ ] `metrics/speed.py` — latency, throughput, VRAM peak
- [ ] `metrics/visual.py` — FVD, PSNR, SSIM via `torchmetrics`
- [ ] Results → JSON; a basic comparison/plotting script for multiple JSONs

**Done when:** you have a real baseline number for Cosmos-Predict 7B on your hardware across all four horizons, in a JSON you can diff later.

---

## Phase 3 — Physics metrics

**Goal:** the axis that makes this project different from every other benchmark gets wired in.

- [ ] Clone PAI-Bench and WorldRoamBench, get each running standalone first (expect rough research-code edges — don't debug your integration and their bugs at the same time)
- [ ] `metrics/physics.py` — clean wrapper, frames → score dict
- [ ] Plug into the Phase 2 runner so every run outputs speed + visual + physics together

**Done when:** a single runner invocation prints all three axes for one model/horizon.

**Why now and not later:** this is the most likely place to hit multi-day dependency hell. Hitting it right after Phase 2 (not after building three optimization modules on top of an incomplete runner) means you find out early, not in week 10.

---

## Phase 4 — PAES metric

**Goal:** turn the three axes into one comparable number.

- [ ] Implement `compute_paes(speedup, physics_score, drift_rate, horizon_t)` in `metrics/paes.py`
- [ ] Run it on the real baseline numbers from Phase 2/3 — no optimization applied yet, so this is just wiring, not tuning
- [ ] Sanity-check the formula against intuition once you have a second data point (Phase 5)

**Done when:** every benchmark run auto-reports a PAES score.

---

## Phase 5 — Optimization stack base + WorldCache

**Goal:** first real optimization, and proof the plug-in architecture works.

- [ ] `optimizations/base.py` — `OptimizationModule` abstract class with `supported_architectures`
- [ ] `stack.py` — `OptimizationStack`: filters incompatible modules by architecture, chains the rest
- [ ] `optimizations/worldcache.py` — wrap the published WorldCache repo's caching logic behind your interface
- [ ] Benchmark with vs. without WorldCache → confirm you reproduce their published 2.3x speedup before trusting anything downstream

**Done when:** you have a baseline-vs-WorldCache PAES comparison that matches the literature's speedup number.

---

## Phase 6 — AdaCache + quantization

**Goal:** fill out the comparison table; same pattern as Phase 5, lower risk.

- [ ] `optimizations/adacache.py`
- [ ] `optimizations/quantization.py` (FP8/INT4 via TorchAO — architecture-agnostic, so cheapest module to add)
- [ ] Full comparison table: baseline / WorldCache / AdaCache / quantization, all three axes + PAES

**Done when:** the table in outline §3.2's example output is real, not illustrative.

---

## Phase 7 — Drift correction (the novel research piece)

**Goal:** the actual contribution. Do this only once Phases 1–6 are stable — you need a working benchmark and a working WorldCache integration to even know if drift correction helps.

- [ ] Read WorldCache's curvature-scoring source thoroughly first
- [ ] Optical-flow-divergence drift monitor
- [ ] Selective recomputation of top-K curvature tokens on threshold crossing
- [ ] Blend corrected tokens back into cached state
- [ ] Benchmark: does it recover 60–80% of the physics gap while keeping ~WorldCache's speedup? That's the target from outline §3.5.

**Done when:** you have a Pareto plot showing drift-correction-on-top-of-WorldCache beating plain WorldCache on PAES.

**Fallback (already agreed in the outline):** if this doesn't pan out in time, Phases 1–6 alone are a publishable benchmark + metric contribution. Don't let this phase block shipping.

---

## Phase 8 — `worldserve`

**Goal:** productize. Last, because it just wraps the stack that now exists.

- [ ] `WorldModelServer` class — loads model + stack once, holds in memory
- [ ] FastAPI `/generate` endpoint, returns frames + PAES profile
- [ ] Async request queue + batching
- [ ] Dockerfile + CLI (`worldserve start --model ...`)

**Done when:** you can `docker run` the server and hit it over HTTP.

---

## Quick reference — order at a glance

```
0. Get access + compute + repo scaffold running
1. WorldModelInterface + Cosmos-Predict          ← one model, works end to end
2. Benchmark runner (speed + visual)              ← measure before optimizing
3. Physics metrics                                ← the differentiator, do it early
4. PAES metric                                    ← wiring, not tuning
5. WorldCache module                              ← first optimization, reproduce known result
6. AdaCache + quantization                        ← fill out the table
7. Drift correction                               ← the hard novel part, only once 1-6 are solid
8. worldserve                                     ← product wrapper, last
```

This matches the Build Order already in `world_model_project_outline.md` §7 — this doc just breaks each step into a checklist with a concrete "done" condition, so it's easier to work off day to day.
