# Outline status: what the project outline asks for, and what exists

Section-by-section map from `world_model_project_outline.md` to the repo, written 2026-10-04 by checking the code, not from memory. **Built** = implemented and measured or tested; **Partial** = something real, but not what the outline literally says; **Not built** = with the reason. Detailed results live in the documents named in each row. "Untested" means written but never run against the real thing.

## 3.1 WorldModelInterface and the v1 models

| Item | Status |
|---|---|
| `WorldModelInterface` (`generate`, `get_info`), with Image2World / Video2World conditioning (12-G) | **Built.** `generate(prompt, init_frame, init_video, actions, horizon, **kwargs)`; a `Rollout` with frames, fps, metadata. |
| Dreamer (DreamerV3, DMC walker) | **Built and the best-measured model.** Trained locally (115k and 515k steps), simulator ground truth. `DREAMER_SETUP.md`. |
| Wan2.1-1.3B (listed under future work) | **Built** (not a world model, a text-to-video model; used for the diffusion work). `DESIGN_DIFFUSION.md`. |
| Cosmos-Predict (2B / "7B" / 2.5-2B) | **Untested.** A wrapper exists (`models/cosmos_predict.py`) but cannot run: there is no Hugging Face token on this machine and the `nvidia/Cosmos-Predict2*` repos return 401 until the license is accepted. The outline's own note that checkpoint sizes are 2B and 14B, not 7B, is unresolved. Needs the user. |
| Open-Sora | **Not built.** Needs its own codebase and several GB of weights; Wan2.1 covers the "open diffusion model" role. |
| V-JEPA 2 (and the 12-B "joint scale across architectures" claim) | **Not built.** V-JEPA 2 predicts representations, not pixels: there is no decoder to turn its predictions into frames, so it cannot be scored by the frame-based harness. Putting a JEPA model on the same speed / visual / physics scale needs one with a decoder. |
| Genie 3 | **Not built.** API closed (the outline says so). |

## 3.2 Benchmark suite

| Item | Status |
|---|---|
| Speed: latency, fps, GPU memory | **Built.** Peak and held (graphs' persistent pools) memory. |
| Speed: time to first frame | **Built, trivially:** equals the latency for every wrapper here (they return when done); a streaming model can report a smaller value in its Rollout metadata. |
| Visual: PSNR, SSIM, temporal consistency | **Built.** |
| Visual: FVD | **Partial.** A Frechet distance in DINOv2 space (`metrics/fvd.py`, `scripts/analyze_fvd.py`), clearly not standard I3D FVD and not comparable with published values. Plus reference-free statistics (`metrics/blind.py`) and learned DINO / CLIP scores (`metrics/perceptual.py`) that the outline does not mention. |
| Physics: PAI-Bench, WorldRoamBench | **Not built** (explicit `NotImplementedError` stubs). Neither has a documented standalone API, and both score with large VLM judges. For Dreamer the physics score is simulator fidelity as a skill score instead; text-to-video has no physics score here, so Wan is judged by agreement with its own unoptimized output. |
| Long-horizon drift curve at 4 / 16 / 60 / 120 s | **Partial.** Dreamer drift is measured at 2-20 s on held-out walking (`DREAMER_SETUP.md`). 60 s and 120 s cannot have ground truth on this task (episodes end at 25 s); speed and memory at 60 / 120 s were measured (linear scaling, `DREAMER_SETUP.md`). |
| The per-run output block | **Built.** `reporting.format_profile`. Drift onset prints as "n/a (not computed)". |
| Comparison script and Pareto plots (7.2 step 5) | **Built.** `scripts/plot_pareto.py`, `compare_diffusion.py`, `analyze_perceptual.py`, `reporting.plot_pareto`. |

## 3.3 PAES

**Built.** `metrics/paes.py`, decay-only drift by default (the outline's own `abs` version is kept as `drift_mode="abs"`). For text-to-video the "physics" term is fidelity to the baseline, not physics, and every report says so.

## 3.4 Optimization stack

| Item | Status |
|---|---|
| Plug-in modules, architecture filtering, registry | **Built.** 31 modules with declared requirements, maturity, exclusive groups; `library_report`; `OptimizationStack` skips what does not apply with a reason. |
| `constraints={min_physics_score, target_speedup, max_vram_gb}` | **Built, as checks and a search, not a predictor.** `constraints.py`; `OptimizationStack(..., constraints=...).check(results)`; `autotune(..., constraints=...)`. |
| `worldcache`, `adacache` | **Built from the papers' descriptions** (no code was available) and measured. WorldCache's 2.3x-at-99.4%-quality is **not reproduced** (different metric and model). AdaCache is dominated by WorldCache on Wan. |
| `fp8_quantization` | **Built** (TorchAO); slower than the baseline on Dreamer. |
| `drift_correction` | **Partial**, see 3.5. |
| `kv_cache_ar` (future) | **Not built:** no autoregressive model. |

## 3.5 Drift correction

**Partial.** The outline itself says the original design overlaps WorldCache and lists three rescopings. Built: a portable guard under a cache (12-A1), measured on AdaCache (`DESIGN_DIFFUSION.md`): it helps AdaCache at matched speed and does not catch WorldCache. **Not built:** token-level selective recompute (the original 3.5 mechanism; it needs partial recompute through attention), quantization-induced drift correction (12-A2; quantization was slower on Dreamer, so there is nothing to correct for speed's sake), autoregressive drift correction (12-A3; no AR model). The outline's "recover 60-80% of the physics gap" target is untestable: Wan has no physics score.

## 3.6 `worldserve`

| Item | Status |
|---|---|
| Server holding model and stack, async queue, batching, CLI, Docker | **Built, with one substitution and one gap.** `worldoptbench/serve.py`: a standard-library HTTP server (the outline says FastAPI; FastAPI is not installed and the stdlib server needs no dependency), bounded queue, micro-batching, result cache, `worldserve` CLI, `from worldserve import WorldModelServer`, `server.serve(port=...)`. Run end to end against real Wan. The Dockerfile is **untested** (no container runtime here). |
| `stack=`, `physics_budget`, `speed_budget`, `hardware` arguments | **Partial.** `stack=` and `constraints=` (the speed target can be checked while serving when a baseline is known; the physics and VRAM limits are reported as unverifiable per request). No `hardware=` argument. |
| "Returns the PAES profile with every generation" | **Not built.** PAES needs a baseline and a reference, which an arbitrary request does not have. Each response carries latency, queue time, batch size and, when a baseline is known, speedup. |

## 4 Future work

WAN2.1: **built**. Fine-tuning / SFT: **not built, by the outline's own scoping** (it would change the project's training-free positioning). AR modules (KV cache, speculative decoding), JEPA modules, autoscaling, managed API, LLM sidebar: **not built** (no model, no hardware, or out of scope). Domain-specific physics metrics (contact dynamics, trajectory plausibility): **not built**; Dreamer's simulator skill score is the only domain metric.

## 12 Open research directions

| Item | Status |
|---|---|
| A. Drift correction pivot | **Partial**, see 3.5. |
| B. Cross-architecture "first" claim (JEPA) | **Not attempted** (see V-JEPA above). |
| C. The framework claim | Unchanged; nothing to build. |
| D. Functional-utility preservation (a 4th axis) | **Built and measured on Dreamer** (`worldoptbench/utility.py`, `models/dreamer_policy.py`, `scripts/policy_ranking_dreamer.py`, results in `DREAMER_SETUP.md`): no optimization measurably changed how the model ranks policies, including one that destroyed its physics score and its absolute returns. One model, one task, nine policies per set; the test cannot see small losses. Not run on a diffusion model. |
| E. Cost-aware Pareto and hardware recommender | **Partial.** Dollars per generated second and the cheapest configuration above a quality floor (`cost.py`, `scripts/cost_pareto.py`) for a user-supplied hourly rate. There is no hardware recommender: latency was measured on one GPU, and translating it to other GPUs would be a guess. |
| F. Public leaderboard | **Partial.** `LEADERBOARD.md`, generated from the result files by `scripts/make_leaderboard.py`: a static snapshot from one machine, not a service. |
| G. Interface completeness (image / video conditioning) | **Built** (see 3.1). |

## Where each result lives

`LEADERBOARD.md` (tables), `DREAMER_SETUP.md` (Dreamer environment, sweeps, long horizons, policy ranking), `DESIGN_DIFFUSION.md` (Wan sweeps, quality measures, WorldCache, AdaCache, the guard), `OPTIMIZATION_CATALOG.md` (every candidate optimization and its status), `TEST_COMMANDS.md` (what to run; the tests have not been run by the assistant that wrote them, only checked through a stub).
