# WorldOptBench — Build Plan

> Companion to `world_model_project_outline.md` (the spec/pitch doc). That doc says *what* everything is. This doc says *what order to actually build it in* and *what "done" looks like at each step*, so you always know what to open your editor and work on next.

**Guiding principle:** get one model generating frames end-to-end as fast as possible (even if ugly), wrap it in measurement before you touch any optimization, then layer optimizations from safest/most-proven → hardest/most-novel, and build the server last because it depends on everything above being stable. Don't build worldserve or drift correction early — there's nothing to serve or correct yet.

---


## Where we are (2026-10-03) — summary; phase detail below is the history

| Phase | Reality |
|---|---|
| 0 Setup | Done for **Dreamer** (Python venv, CUDA 12.8 torch for the RTX 5070 Ti, TensorRT, a trained walker). Cosmos access is **not** done and is no longer on the critical path. |
| 1 Interface + model | `WorldModelInterface` done; **Dreamer wrapper working** (`models/dreamer.py`). Cosmos wrapper exists but is untested. |
| 2 Benchmark runner | Done and trustworthy after methodology fixes (scenarios built up front, per-horizon warm-up, timing separated from metrics, repeated runs with medians). |
| 3 Physics | Simulator-fidelity skill score done; PAI-Bench / WorldRoamBench wrappers still stubs (only relevant to text-to-video models). |
| 4 PAES | Done; drift penalty is now decay-only. Drift not yet measurable (needs a better-trained model and `prompts/dreamer_dmc_set_10seeds.json`). |
| 5-6 Optimization stack | Stack + 11 built optimizations; CUDA graphs and TensorRT help, quantization/precision do not. WorldCache built and measured on Wan2.1-1.3B (see DESIGN_DIFFUSION.md; the paper's 2.3x-at-99.4% claim is **not** reproduced; its motion warping is implemented and measured as a no-op); AdaCache built and measured (dominated by WorldCache on this model; see DESIGN_DIFFUSION.md). Learned perceptual scores (DINOv2 + CLIP) added and used to rank the diffusion configurations (WorldCache clearly ahead at 1.65-2x). Chunked rollouts (`DESIGN_STEP_HOOK.md` step 1) built and measured: equivalent output, ~0.2 ms per chunk boundary, not a speedup. |
| 7 Drift correction | Rescoped per outline §12-A1 and built as a portable guard for AdaCache (`guard=`): helps AdaCache at matched speed, does not beat WorldCache; token-level selective recompute and the physics-gap target not attempted (see DESIGN_DIFFUSION.md). |
| 8 worldserve | Built (`worldoptbench/serve.py`, stdlib HTTP, queue + micro-batching + cache + `worldserve` CLI), unit-tested with fake models and run end to end against real Wan2.1 over HTTP (8.9 s per 2 s clip with the recommended stack, cache hit 0.02 s); Dockerfile written but **untested** (no container runtime here). |
| Beyond the plan | Constraints (`constraints.py`), cost per generated second (`cost.py`), leaderboard (`LEADERBOARD.md`), DINO-space Frechet distance, long-horizon scaling on Dreamer, and the §12-D policy-ranking check (optimizations preserve policy ranking within noise). See `OUTLINE_STATUS.md` for what the outline asks for that was not built. |

**Done since:** the long run finished (515k env steps, eval return ~790). On it: speed ranking reproduced; 21 optimizations measured (best: `cuda_graphs` + TensorRT fp16, ~3.5-9.9 ms vs ~210 ms eager); **drift measurable (-0.092 +- 0.009 skill/s) on held-out walking at short horizons** but not on random-action scenarios (those measure a falling walker, and long horizons are saturated: both models score ~0.02 skill at 5 s on competent walking); `autotune` run end to end and agreed with the sweeps. **Next:** per-step hook + chunked graphs (`DESIGN_STEP_HOOK.md`); `worldserve`; Cosmos access to unlock the 42 diffusion items. Candidate list: `OPTIMIZATION_CATALOG.md`.

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

**Dreamer as a no-Hugging-Face first model (added 2026-10-03):**

- [x] `models/dreamer.py` — `DreamerWorldModel` (imagination rollout from context frames + actions), `DreamerRepo` (loads a clone of NM512/dreamerv3-torch: config, env, spaces), `DreamerSimScenarios` (steps the real simulator for context frames, actions, and **ground-truth** continuation). Written from the repo source, **not run** — see the module docstring's INFERRED list.
- [x] `scenarios.py` + `run_benchmark(scenario_fn=...)` — lets a model take more than a prompt and supply ground truth, with scenario building kept outside the timed region
- [x] `metrics/physics.compute_sim_fidelity` — physics score from simulator ground truth (1 − mean normalized pixel error + per-step error curve). Used instead of PAI-Bench when reference frames exist; PAES/drift pick it up automatically. Coarse (see its docstring) — revisit with real numbers
- [x] `prompts/dreamer_dmc_set.json` — horizons 2/5/10/20 s, not 4/16/60/120: a DMC episode is ~25 s, so there's no ground truth beyond that
- [x] Environment built (Python 3.11 venv, CUDA 12.8 torch for the 5070 Ti, dreamerv3-torch cloned + patched) — see `DREAMER_SETUP.md`. First real run of `DreamerWorldModel`/`DreamerSimScenarios` against a 5k-step smoke checkpoint worked with **no code fixes needed**: env wrapper attribute forwarding, `env.step` shape, and the 20 steps/s assumption all held
- [x] `compute_sim_fidelity` upgraded to a skill score vs a "nothing moves" baseline after the real run showed the raw pixel score couldn't separate an undertrained model (0.93) from freezing the last frame (0.92). Skill: model ~0.2-0.3, freeze 0.0, perfect 1.0
- [x] Reproducibility: unseeded Dreamer generations from *identical weights* differed ~12/255 with fidelity 0.06-0.27 (noise > any effect). `generate(seed=...)` now seeds the latent sampling; scenarios pass per-scenario seeds
- [x] Real baseline + quantization comparison done on a 115k-step checkpoint — see `DREAMER_SETUP.md` "Baseline results". Headline: quantization is no faster (launch-bound model), physics unchanged; the world model is under-trained (skill ~0.25), so drift/PAES differences aren't meaningful yet
- [ ] Train the world model much longer (resume `logdir\walker` to ~500k steps) before trusting drift numbers; then add more seeds
- [x] `optimizations/cuda_graphs.py` — `CudaGraphsModule` / `CudaGraphExecutor`: replays the imagination loop as one CUDA graph via a new `tensor_executor` extension point (`models.base.HasTensorExecutor`). **5.5x mean speedup (2.6x at 2 s -> 8.6x at 20 s), bit-identical output.** Used raw CUDA graphs, not `torch.compile` (needs Triton + a C toolchain on Windows). Context phase stays eager (the repo's `obs_step` host-syncs). First module that actually makes the model faster; quantization was a net loss.
- [x] `vram_peak_gb` flattered graphs (missed persistent pools): added `SpeedMetrics.vram_reserved_gb` (memory held after the call), which shows graphs cost +128 MB
- [x] More Dreamer optimizations built and swept (`scripts/sweep_dreamer.py`): `precision` (bf16/fp16 autocast), `tf32`, `lean_scan`, and the context phase fused into the CUDA graph (graphs 5.5x -> ~11.5x). Full table in `DREAMER_SETUP.md`. Only graphs (and now TensorRT) matter; lean/tf32 showed no effect once repeated
- [x] **TensorRT** (`optimizations/tensorrt_backend.py`) + `gumbel_sampling` + `no_dist_validation`, via a new `imagine_backend` hook on the model. ONNX-exports one RSSM step, builds a TensorRT 11.3 engine (cached on disk), runs it per step inside the CUDA graph. **Best configuration: 13.2 ms mean, 1.64x faster than the repo loop in graphs, 1.33x faster than the pure-PyTorch step in graphs**, matching its PyTorch reference to 2.5e-6 in skill. Uses Gumbel-max sampling (statistically equivalent to `torch.multinomial`: pooled skill 0.274 vs 0.276, n=180 each). Do NOT install `torch-tensorrt` next to torch 2.11 (it replaces torch).
- [x] Runner methodology fixes found while chasing this: build all scenarios before timing, warm up every horizon, time all rollouts back to back and compute metrics afterwards (metrics between timed calls slowed TensorRT-in-graph calls by ~5 ms), and the sweep now repeats each config 3x with medians + a noise column (the eager baseline varies +-34%). **Earlier 5.5x / 11.5x CUDA-graph figures are superseded (~9x)** — see `DREAMER_SETUP.md`.
- [x] **Library layer (2026-10-03):** modules now declare `requires` / `needs_cuda` / `needs_packages` / `maturity` / `summary`; `OptimizationStack` skips non-applicable modules with a reason (so `available_modules()` can be handed over wholesale); `library_report(model)` lists what applies; `autotune` greedily builds the best stack by measured PAES with a noise margin and a physics-loss guard; `OptimizationStack.restore()` undoes process-global modules. New experimental modules: `cudnn_benchmark`, `channels_last`, `low_rank`, graph-pool sharing, and the VRAM/dtype advisor (`memory.py`). **All unit-tested only; none swept on hardware yet** — sweep them after the long training run finishes.
- [ ] Not feasible here: `torch.compile` (no MSVC/Triton on this machine); custom CUDA kernels (no nvcc)
- [ ] Remaining cost: host<->device copies + Python per call (~20 ms floor at short horizons); batching several rollouts per call is the next lever
- [ ] Caveat to resolve: `compute_paes` penalizes |drift_rate|, so a model whose skill score *rises* with horizon (seen with the undertrained one) is penalized like one that decays. Probably want signed or decay-only drift
- V-JEPA 2 deliberately **not** added as a model under test: it predicts latents, not frames, so visual/physics metrics don't apply. Possible later as a feature extractor for an embedding-based consistency score.

**Skip for now:** Open-Sora, Genie 3, V-JEPA 2 — one model is enough to build everything else against. Add the second model only once the benchmark runner (Phase 2) exists, as a test that your abstraction is actually architecture-agnostic and not secretly Cosmos-shaped.

---

## Phase 2 — Benchmark runner (speed + visual only)

**Goal:** a script that takes any `WorldModelInterface` and produces a results JSON. No physics yet — that's Phase 3.

- [x] `prompts/standard_set.json` — 6 prompts (robotics/AV/indoor + 3 physics-stress prompts: fluid, rigid collision, wind) at 4s/16s/60s/120s horizons
- [x] `runner.py` — calls `.generate()`, times it + tracks VRAM via `metrics/speed.py`, computes visual metrics, writes JSON. Also opportunistically calls Phase 3/4 metrics if they're wired up, degrading gracefully (`NotImplementedError` caught) if not — so this same runner keeps working as later phases land
- [x] `metrics/speed.py` — latency, throughput, VRAM peak (VRAM degrades to `None` without CUDA, e.g. this dev laptop)
- [x] `metrics/visual.py` — temporal consistency (no reference needed, always computed); PSNR/SSIM via `torchmetrics` **only if `reference_frames` is supplied** — pure Text2World prompts have no ground truth to compare against, so these are `None` by default (see caveat added to outline §3.2)
- [x] `reporting.py` — `load_results`/`compare`/`summarize` (no extra deps) + `plot_pareto` (needs the `plot` extra)
- [x] Tests: `test_speed.py`, `test_visual.py`, `test_runner.py`, `test_reporting.py` — all pass without GPU/torch (33 passed, 1 skipped for the torchmetrics-only test)

- [x] Pre-Phase-5 runner fixes: `RunResult.optimization` label, `baseline_path=` (loads a prior results JSON for speedup instead of hand-built dicts), and per-prompt drift fitting — physics is scored at every horizon first, then `compute_drift_curve` runs across them, then PAES gets that drift rate (previously `drift_rate` was always 0)

**Not done — real number still pending Phase 0:** everything above is boilerplate verified against a `FakeWorldModel`, not against Cosmos-Predict on real hardware. That's still the actual "done" condition.

**Done when:** you have a real baseline number for Cosmos-Predict on your hardware across all four horizons, in a JSON you can diff later.

---

## Phase 3 — Physics metrics

**Goal:** the axis that makes this project different from every other benchmark gets wired in.

- [x] `metrics/physics.py` scaffolded: `PhysicsScore`/`DriftCurve` dataclasses, and `compute_drift_curve` — real, tested, pure-numpy curve fitting over scores-by-horizon (doesn't depend on either external repo)
- [ ] `compute_pai_bench_score` / `compute_worldroambench_score` — left as explicit `NotImplementedError` stubs. Checked: **neither PAI-Bench (`SHI-Labs/physical-ai-bench`) nor WorldRoamBench has a documented standalone Python API** as of 2026-09-12 — no point guessing function names that'd silently do the wrong thing. Clone both, find their real entrypoints, fill these in for real
- [x] Runner already wired to call these opportunistically (Phase 2) — filling in the two stubs above is the only thing needed to make full profiles start flowing, no runner changes required
- [ ] Clone PAI-Bench and WorldRoamBench, get each running standalone first (expect rough research-code edges — don't debug your integration and their bugs at the same time)

**Done when:** a single runner invocation prints all three axes for one model/horizon.

**Why now and not later:** this is the most likely place to hit multi-day dependency hell. Hitting it right after Phase 2 (not after building three optimization modules on top of an incomplete runner) means you find out early, not in week 10.

---

## Phase 4 — PAES metric

**Goal:** turn the three axes into one comparable number.

- [x] `compute_paes(speedup, physics_score, drift_rate, horizon_t)` in `metrics/paes.py` — implemented, tested (4 tests: no-drift baseline, speedup increases score, drift penalizes score, sign of drift_rate doesn't matter)
- [x] Runner already calls it automatically once `physics.pai_bench_score` is available (Phase 3) — nothing left to wire
- [ ] Run it on the real baseline numbers from Phase 2/3 — no optimization applied yet, so this is just wiring, not tuning. **Blocked on Phase 0/3 real data**
- [ ] Sanity-check the formula against intuition once you have a second data point (Phase 5)

**Done when:** every benchmark run auto-reports a PAES score.

---

## Phase 5 — Optimization stack base + WorldCache

**Goal:** first real optimization, and proof the plug-in architecture works.

- [x] `optimizations/base.py` — `OptimizationModule` abstract class with `supported_architectures`, plus a name registry. Modules take and return a `WorldModelInterface` (not a raw pipeline) — see its docstring for why; WorldCache will need a way for the model wrapper to expose its loaded pipeline(s), to be designed once its source is readable
- [x] `stack.py` — `OptimizationStack`: filters incompatible modules by architecture (recorded in `.skipped`), chains the rest, `.name` feeds `run_benchmark(optimization=...)`. Constraint solver deferred. Note `ModelInfo.supported_optimizations` is NOT used for filtering (Cosmos doesn't list every module it can take) — consider dropping it
- [ ] `optimizations/worldcache.py` — wrap the published WorldCache repo's caching logic behind your interface
- [ ] Benchmark with vs. without WorldCache → confirm you reproduce their published 2.3x speedup before trusting anything downstream

**Done when:** you have a baseline-vs-WorldCache PAES comparison that matches the literature's speedup number.

---

## Phase 6 — AdaCache + quantization

**Goal:** fill out the comparison table; same pattern as Phase 5, lower risk.

- [ ] `optimizations/adacache.py`
- [x] `optimizations/quantization.py` — `QuantizationModule(scheme=...)` over TorchAO (int8/int4/float8 weight-only, int8/float8 dynamic), Linear layers only, registered as `"quantization"`. Works on any model with `torch_module()` (`models.base.TorchBacked`) — implemented by `DreamerWorldModel` so far, not Cosmos. Verified on the Dreamer model with TorchAO 0.18.0: int8/float8 weight-only and int8/float8 dynamic work (13 Linear layers each); int4 needs `mslk`. Seeded outputs differ from fp32 by only ~1.3/255, skill 0.23-0.26 vs 0.253 fp32 (undertrained smoke checkpoint, so just plumbing). Refuses to run if zero layers match so a no-op can't be reported as quantized. Expect small models like Dreamer's to show speedup < 1 — that's a legitimate result
- [ ] Run it: baseline Dreamer → `OptimizationStack(model, ["quantization"], module_kwargs={"quantization": {"scheme": "int8_weight_only"}})` → compare via `run_benchmark(baseline_path=...)`
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
