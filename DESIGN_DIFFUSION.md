# Design: diffusion world models (Wan2.1 now, Cosmos later)

Status: **design only; nothing below is implemented.** Written 2026-10-03. Every API fact here was read from the installed `diffusers 0.40.0` source, not recalled — the Cosmos wrapper written earlier was a pile of unverified guesses, and the lesson was to build against something we can run.

## Why Wan2.1-T2V-1.3B instead of waiting for Cosmos

- **Runs here.** Public (Apache 2.0), ships as a `diffusers` pipeline (`WanPipeline`), ~1.3B-parameter transformer (~2.7 GB in bf16), so it fits a 12 GB GPU. Cosmos-Predict2 is gated (`"gated": "auto"`, NVIDIA Open Model License click-through) with a `cosmos` library and a `diffusers` integration.
- **Same architecture class.** Both are video DiTs with flow-matching samplers, so caching, token/attention, step-reduction and guidance optimizations transfer. Swapping Cosmos in later should be a wrapper change plus re-measurement, not a redesign. (Not guaranteed: Cosmos-Predict2 is action/video-conditioned and its blocks differ in detail.)
- **Honest limits.** Wan is a text-to-video model, not a world model. It exercises the *diffusion optimization* machinery; it says nothing about physics. See "Evaluation".

## What diffusers already provides (verified in the source)

`diffusers.hooks` exports `FirstBlockCacheConfig`, `MagCacheConfig`, `FasterCacheConfig`, `PyramidAttentionBroadcastConfig`, `TaylorSeerCacheConfig`, `LayerSkipConfig`, `TextKVCacheConfig`, `SmoothedEnergyGuidanceConfig`, `apply_group_offloading`, `apply_layerwise_casting`, `apply_tensor_parallel`, `apply_context_parallel`, plus `HookRegistry` / `ModelHook` for writing our own. `WanTransformer3DModel` is a `CacheMixin` (`enable_cache(config)`, `disable_cache()`, `cache_context()`), has `set_attention_backend(...)` (backends include `native`, `_native_cudnn`, `_native_efficient`, `flex`, `flash`, `sage`, `xformers`), `enable_layerwise_casting`, `enable_group_offload`, `compile`.

Mapping to `OPTIMIZATION_CATALOG.md` (so these are *wrap-and-measure*, not build-from-scratch):

| Catalog | diffusers feature |
|---|---|
| E3 TeaCache-style | `FirstBlockCacheConfig`, `MagCacheConfig` |
| E4 FasterCache | `FasterCacheConfig` |
| E5 Pyramid Attention Broadcast | `PyramidAttentionBroadcastConfig` |
| E6 TaylorSeer | `TaylorSeerCacheConfig` |
| E7 block/layer skipping | `LayerSkipConfig` |
| E10 text / cross-attention K/V cache | `TextKVCacheConfig` |
| F1-F3, B14 attention kernels | `set_attention_backend(...)` (availability on Windows varies: `flash`/`sage` need extra packages) |
| J2 offloading | `apply_group_offloading` |
| B-series fp8 *storage* | `apply_layerwise_casting` |
| K1 / K3 | `apply_tensor_parallel` / `apply_context_parallel` (multi-GPU only: not testable here) |
| D1 better solvers | swap `pipe.scheduler` (diffusers ships UniPC, DPM-Solver, etc.) |
| D3 CFG truncation | set guidance off partway via `callback_on_step_end` (the pipeline reads `guidance_scale`) |

**Not in diffusers, would be ours:** WorldCache, AdaCache, X-Cache-style region reuse (E1, E2, E9), the sparse-attention family (F4), speculative decoding for diffusion (G1), drift correction (M1), step distillation (D7, needs training).

## The pipeline structure the hooks must fit

`WanPipeline.__call__` denoises in a Python loop. Per step it calls the transformer **twice** (a conditional pass under `cache_context("cond")` and, with classifier-free guidance, an unconditional pass under `cache_context("uncond")`), combines them, then `scheduler.step`. It exposes `callback_on_step_end(pipe, step, timestep, kwargs)` and `callback_on_step_end_tensor_inputs`. The transformer forward is: patch-embed -> condition embedder -> a loop of `WanTransformerBlock(hidden_states, encoder_hidden_states, temb, rotary_emb)` -> output projection. Consequences:

1. Per-step control (CFG truncation, step skipping, schedule changes) goes through `callback_on_step_end` and pipeline properties, not a new hook system.
2. Per-block control (caching, layer skipping) goes through `HookRegistry` on the transformer blocks — which diffusers' own cache configs already use.
3. The two CFG passes have separate cache states, which is why `cache_context` exists; any cache we write must respect it.

So the `DENOISE` hook from the catalog is not a new framework: it is "a diffusion model wrapper that exposes its pipeline and transformer, and modules that call diffusers' hook APIs".

## Plan (each stage is verified on the real model before the next)

1. **Wrapper:** `WanVideo` implementing `WorldModelInterface` (`architecture="diffusion"`). Prompts are encoded **once** and cached (the UMT5-XXL text encoder is ~22 GB in fp32 and does not fit beside the transformer on 12 GB; encode on CPU, save embeddings, drop the encoder). Baseline run end to end with fixed seeds; record latency, peak VRAM, frames.
2. **Cache modules**, one per diffusers config, each with `requires=("transformer",)`-style declaration and a unit test on a *tiny random-weight DiT* (CPU, no download) for the properties that don't need the real model: caching disabled == baseline exactly; the unconditional and conditional passes keep separate state; reset between videos.
3. **Measure on the real model:** speed *and* fidelity, on short clips first (e.g. 33 frames at 320x192) because full 480p/81-frame videos take minutes each on this GPU. A handful of full-size runs for headline numbers.
4. **Then ours:** WorldCache / AdaCache implemented as `ModelHook`s, compared against the diffusers ones.
5. **Then Cosmos:** a `CosmosPredict` wrapper against the same interface, once access is granted.

## Evaluation (the part to be careful about)

- **No physics score exists for Wan here.** PAI-Bench / WorldRoamBench wrappers are still stubs and score world-model plausibility, not this.
- **Fidelity to the unoptimized output** is the standard way caching papers report quality: same prompt, same seed, same noise, compare the optimized video to the baseline's with PSNR / SSIM / LPIPS. That is what `physics` will be for diffusion until a real physics scorer exists, and PAES must say so wherever it is reported: it measures *agreement with the baseline*, not correctness.
- **Seeds and noise.** Optimizations that don't change the noise are compared on identical seeds (paired, tight). Anything stochastic needs many seeds, and the effective sample size is the number of distinct prompts/seeds, not frames (the same lesson as the Dreamer sweeps).
- **Speed:** repeated runs and absolute latencies, GPU idle, timing separated from metric computation, per the runner's existing methodology.

## Known risks

- 12 GB VRAM is tight: transformer bf16 (~2.7 GB) + activations at 480p x 81 frames, plus the VAE (use tiling). May need to run smaller clips; some caches add memory.
- Windows: `flash` and `sage` attention backends need packages that may not build here; `torch.compile` is unavailable (no Triton/MSVC), so `compile` is out. Expect some catalog items to be marked blocked-by-toolchain.
- Download size: the repo is ~43 GB (fp32). The transformer, VAE, tokenizer and scheduler are a few GB; the text encoder is the bulk.
- A cache that works on Wan may not transfer to Cosmos unchanged; the Cosmos stage re-measures everything.

## Measured results on Wan2.1-T2V-1.3B (stage 3)

Setup: 4 prompts x 1 seed, 2 s clips (33 frames) at 192x320, 30 steps, CFG 5.0, RTX laptop GPU (12 GB), one process per configuration, GPU otherwise idle. Baseline 12.4 s per clip (two baseline runs agreed). "Fidelity" is agreement with the *unoptimized video for the same seed*: PSNR (dB), SSIM, and a skill score against repeating the first frame (0 = no better than a frozen video, 1 = identical). It is not physics or prompt adherence. n = 4 clips per row, one run each: treat small differences as noise.

| configuration | latency | speedup | PSNR | SSIM | skill | note |
|---|---|---|---|---|---|---|
| cfg_trunc_0.6 | 10.2 s | 1.22x | 30.3 | 0.970 | 0.62 | best quality per speed so far |
| cfg_trunc_0.4 | 9.1 s | 1.37x | 25.0 | 0.924 | 0.40 | |
| pab_2 | 11.5 s | 1.08x | 25.8 | 0.935 | 0.43 | |
| pab_3 | 11.4 s | 1.09x | 23.8 | 0.893 | 0.28 | |
| first_block_cache 0.05 | 10.2 s | 1.22x | 18.6 | 0.809 | 0.19 | dominated by cfg_trunc_0.6 at equal speed |
| first_block_cache 0.1 / 0.2 | 7.1 / 4.8 s | 1.76x / 2.62x | ~10 | ~0.45 | 0 | fast but the video is no longer the same video |
| taylorseer 3 / 5 | 9.3 / 8.7 s | 1.34x / 1.43x | 12.2 / 11.3 | 0.59 / 0.53 | 0 | after the pattern fix (see below) |
| steps 20 / 15 | 8.7 / 6.8 s | 1.43x / 1.83x | 11.0 / 10.5 | ~0.46 | 0 | see caveat |
| layer_skip 2 / 4 | 11.7 / 11.0 s | 1.06x / 1.14x | ~10 | ~0.4 | 0 | wrecks quality for little gain |
| fp8 weight storage | 13.7 s | 0.91x | 13.2 | 0.64 | 0.05 | saves ~1.3 GB peak (4.4 -> 3.1), costs speed |
| attention native / efficient | 12.5 / 12.4 s | 1.0x | identical | | | same kernel in practice |
| attention cudnn | 11.4 s | 1.10x | 15.4 | | | output differs: a numerically different kernel, which is enough to hit the metric's noise floor (see the floor section below) |
| attention flex | 36.4 s | 0.34x | 16.5 | | | slower here |
| attention flash (SDPA) | | | | | | unavailable on this machine ("No available kernel") |

Caveats that change how to read this:

- **Fewer steps and any other change that perturbs the trajectory scores near zero skill** even when the video may be perfectly good, because the metric asks "is this the same video", not "is this a good video". It penalizes the step-reduction family structurally. A quality metric that doesn't need the baseline (or LPIPS/VBench-style) is needed before concluding those are bad.
- Only cfg_trunc and PAB kept the video close to the baseline; everything else with a real speedup changed it substantially.
- Peak memory is per process (model load included); the earlier single-process sweep reported ~7.5 GB because the reference generator was alive in the same process.

Bugs found by measuring (all fixed, with tests): TaylorSeer's default diffusers patterns end in `attn` and are full-matched, so on Wan (`attn1`/`attn2`) they hooked nothing and the first measurement (1.00x, identical output) was a silent no-op; the fp8 module's default compute dtype came from `next(parameters())`, which is Wan's fp32 `scale_shift_table`, so the pipeline cast latents to float32 and the bf16 patch embedding rejected them; running all configurations in one process let one failed attention backend break the next four.

Open: combinations (cfg_trunc + PAB is the obvious first); more prompts/seeds for statistical power; the full-size 480p x 81-frame headline runs. (Why `_native_cudnn` changes the output is explained below: it is the metric's noise floor, not a bug.)

## Statistical sweep on Wan2.1-1.3B (32 clips) and WorldCache

Setup: 16 prompts x 2 seeds = 32 clips (`worldoptbench/prompts/wan_set_16.json`; the first 4 prompts are the screening set), 2 s clips (33 frames) at 192x320, 30 steps, CFG 5.0, one process per configuration, GPU otherwise idle. Reported as mean +- standard error over clips. Two baseline runs were identical (deterministic). Latency varies very little run to run, so the speedup error bars are tiny; the fidelity error bars (about +-0.6 to 0.9 dB PSNR) are dominated by which prompts the clips are, so differences of a few dB between configurations are real and differences under ~1 dB are not. Fidelity is agreement with the unoptimized video for the same prompt and seed, not physics or prompt adherence. Raw per-clip results: `results/wan16/`.

| configuration | speedup | PSNR (dB) | SSIM | skill |
|---|---|---|---|---|
| cfg_trunc 0.6 | 1.21 +- 0.00 | 30.8 +- 0.6 | 0.965 | 0.56 |
| cfg_trunc 0.4 | 1.36 +- 0.00 | 25.9 +- 0.6 | 0.919 | 0.35 |
| cfg_trunc 0.4 + pab 2 | 1.43 +- 0.00 | 26.3 +- 0.6 | 0.921 | 0.35 |
| pab 2 | 1.07 +- 0.00 | 26.6 +- 0.6 | 0.928 | 0.37 |
| first_block_cache 0.05 | 1.23 +- 0.00 | 21.5 +- 0.8 | 0.843 | 0.21 |
| first_block_cache 0.1 | 1.72 +- 0.01 | 14.6 +- 0.8 | 0.610 | 0.03 |
| worldcache 0.02 | 1.21 +- 0.01 | 27.0 +- 0.9 | 0.930 | 0.42 |
| worldcache 0.04 | 1.65 +- 0.01 | 19.2 +- 0.6 | 0.788 | 0.14 |
| worldcache 0.08 | 2.40 +- 0.02 | 13.4 +- 0.7 | 0.547 | 0.02 |
| cfg_trunc 0.6 + worldcache 0.04 | 1.98 +- 0.01 | 19.4 +- 0.6 | 0.789 | 0.14 |
| taylorseer 3 | 1.34 +- 0.00 | 14.5 +- 0.6 | 0.614 | 0.04 |
| steps 20 | 1.43 +- 0.00 | 13.3 +- 0.7 | 0.553 | 0.02 |

What it says (PSNR view; the section below adds a reference-free check that changes some of these conclusions):

- **Up to about 1.4x, CFG truncation is the best option**: 1.21x at PSNR 30.8, 1.36x at 25.9. Adding PAB on top gives about 5% more speed at no measurable fidelity cost (1.43x, 26.3).
- **Above that, a cache is needed.** PSNR alone cannot rank the caches below ~21 dB because of the noise floor, so see the next section for the ranking.
- **TaylorSeer and fewer steps** score low on PSNR; the next section shows that for these two the low score is real degradation (blur, flicker), not just "a different video".

### The metric has a noise floor, and a reference-free check that sees past it

Same-seed generation on Wan is chaotic. Adding Gaussian noise (std 1e-4, or 1e-2, relative to unit-variance noise) to the initial latents, a change at the level of bf16 rounding, gives a mean PSNR of **21.8 / 21.4 dB** against the unperturbed video over the 32 clips (3 prompts looked at first ranged 14-20 dB; individual clips vary a lot). Any computation that is not bit-identical lands there, however equivalent its maths: that is why cuDNN attention scores 15 dB (a different kernel, not a bug), and why PSNR cannot separate "a different but equally good video" from "a degraded one" below roughly 21 dB.

So a second, **reference-free** view was added (`worldoptbench/metrics/blind.py`, `scripts/analyze_blind.py`): seven pixel statistics of each video (sharpness, noise, contrast, colourfulness, motion, flicker, temporal jerk), their ratio to the baseline video's, and the **style deviation** = mean |log ratio| (0 = same statistics; 0.69 = a factor of two on average). Each configuration is compared, clip by clip, with the perturbation control. These are pixel statistics, not a quality model: they see blur, noise, lost or jittery motion and brightness pumping, and they do not see content errors (a wrong object, broken anatomy). Full table: `results/wan_blind/analysis.txt`.

| configuration | speedup | PSNR | style deviation | vs noise control (paired) | what shifted |
|---|---|---|---|---|---|
| perturb 1e-4 / 1e-2 (control) | 1.00 | 21.8 / 21.4 | 0.080 / 0.081 | (null) | |
| pab 2 | 1.08 | 26.6 | 0.059 | within | |
| wc 0.02 | 1.21 | 27.0 | 0.045 | **below** (closer to baseline than noise) | |
| cfg_trunc 0.6 | 1.22 | 30.8 | 0.040 | **below** | |
| first_block_cache 0.05 | 1.24 | 21.5 | 0.089 | within | |
| cfg_trunc 0.4 | 1.37 | 25.9 | 0.058 | within | |
| cfg_trunc 0.4 + pab 2 | 1.44 | 26.3 | 0.057 | within | |
| taylorseer 3 | 1.35 | 14.5 | 0.291 | **beyond** (+0.21) | flicker x1.9, motion +20% |
| steps 20 | 1.43 | 13.3 | 0.332 | **beyond** (+0.25) | sharpness -17%, noise -17%, flicker +38% |
| worldcache 0.04 | 1.65 | 19.2 | 0.126 | beyond, small (+0.045 +- 0.018, ~2.5 sigma) | flicker +17%, motion +8%, sharpness +7% |
| first_block_cache 0.1 | 1.73 | 14.6 | 0.296 | **beyond** (+0.21) | sharpness -23%, flicker +27% |
| cfg_trunc 0.6 + worldcache 0.04 | 1.99 | 19.4 | 0.125 | beyond, small (+0.043 +- 0.018) | as worldcache 0.04 |
| worldcache 0.08 | 2.40 | 13.4 | 0.429 | **beyond** (+0.35) | motion +65%, flicker x2.2 |

What this settles:

- **Up to ~1.44x nothing measurable is lost.** CFG truncation, PAB, their combination (1.44x), WorldCache 0.02 and even first-block cache 0.05 are statistically indistinguishable from (or closer to the baseline than) a numerically-equivalent change, on both PSNR and the blind statistics. Of these, CFG truncation 0.6 and WorldCache 0.02 sit below the noise level, and cfg 0.4 + pab 2 is the fastest.
- **WorldCache is clearly better than first-block cache at ~1.7x**: style deviation 0.126 at 1.65x vs 0.296 at 1.73x, a gap of ~0.17 (about 4 standard errors). This is the comparison the PSNR floor could not make, and it holds. **CFG truncation 0.6 + WorldCache 0.04 is the best ~2x point: 1.99x with a small, measurable shift** (flicker about +17%), the same shift as WorldCache alone.
- **The methods fail differently**, which is useful for choosing: first-block cache and fewer steps lose sharpness (smoother video); TaylorSeer and high-threshold WorldCache add flicker and motion (temporal instability).
- Caveats: pixel statistics only (no content errors); the worldcache 0.04 shift is borderline (about 2.5 sigma with 12 configurations compared, so treat as probable, not proven); 32 clips of 2 s at 192x320 with one model.

### Visual check of the style-deviation metric (contact sheets)

`scripts/make_contact_sheets.py` regenerates videos and lays out the baseline, the perturbation control and a set of configurations side by side (`results/contact_sheets/<prompt>_<gentle|aggressive>.png`, six prompts, each row labelled with its measured speedup and style deviation). I viewed four of the twelve sheets (`ball_ramp` gentle and aggressive, `pan_forest` aggressive, `skateboard` aggressive) and compared what I saw with the per-clip numbers. This is one rater (me) on 3-4 prompts; look at the sheets yourself before leaning on any of it.

What held up:

- **Below about 0.10 per clip, the video looks like the baseline's.** All nine such clips I checked (CFG truncation, WorldCache 0.02, CFG + PAB, the perturbation control on two prompts, WorldCache 0.04 on two prompts, first-block cache 0.05) keep the same scene, objects and motion, differing only in small details.
- **The perturbation control produces a different-but-good video.** On `ball_ramp` it rearranges the ramp (style deviation 0.15) and the result is just as coherent: the noise floor is real, and a "different video" is not a "worse video".
- **WorldCache 0.04 and CFG truncation + WorldCache keep the baseline's scene on all three prompts viewed; first-block cache 0.1 changes the scene on all three** (a bench instead of a ramp, a different sky and lamp post). This supports "WorldCache is better than first-block cache at ~1.7x" on visible grounds, not only on the number.

What did not hold up:

- **Above about 0.15 the number does not separate "broken" from "different but fine".** `taylorseer_3` on `ball_ramp` (0.33) is visibly smeared and has lost the scene; `wc_0.04` on `pan_forest` (0.32) and `cfg0.6 + wc_0.04` on `pan_forest` (0.34) look fine. `ada_fast30` on `pan_forest` (0.43) has turned the forest into abstract shapes, yet scores lower than `wc_0.08` on the same clip (0.61), which keeps the forest with artifacts. Part of this is clip dependence: camera-pan prompts score high for good configurations too.
- **The mean hides bimodal failures.** `ada_slow30` (mean 0.318, median 0.18) is near-perfect on `skateboard` (0.08) and collapses into speckle noise on `ball_ramp` (0.49) and `pan_forest` (1.11), and 8 of its 32 clips are above 0.5 against 5 for first-block cache 0.1 (mean 0.296). `scripts/analyze_blind.py` and `compare_diffusion.py` now also report the median, the 90th percentile and the number of clips above 0.5.
- **Fewer steps is a different video, not obviously a blurrier one**: on the `ball_ramp` sheet `steps_20` (0.41) is a coherent, differently composed scene with an odd outline on the ball. The "sharpness -17%" average is real but the viewed clip does not look blurry.

Consequence: use style deviation as a coarse screen. Reliable at the low end (about 0.1 and below means baseline-like), unreliable for ranking configurations above that. The ordering of the aggressive configurations should be taken from the sheets and from the share of clips that break, not from the mean. A learned feature measure (DINO similarity to the baseline video, CLIP match to the prompt) was added next; see the section below.

### Learned perceptual scores (DINOv2 and CLIP)

Added after the contact-sheet check showed pixel statistics cannot rank configurations above ~0.15 (`worldoptbench/metrics/perceptual.py`, `scripts/analyze_perceptual.py`; results in `results/wan_perceptual/` and `results/wan_perceptual.md`). Models: `facebook/dinov2-small` and a safetensors CLIP ViT-B/32 (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`); about 0.7 GB, scoring a clip takes ~0.1 s, computed after all timing. Three numbers per clip: **DINO similarity** to the baseline video at matching frames (is it the same content and layout; 1.0 = identical), the **change in the video's own DINO temporal consistency** against the baseline's (flicker or jitter the baseline lacks), and the **change in CLIP prompt match** against the baseline's. Same 32 clips; the perturbation controls give DINO similarity 0.876 and 0.888, and the 5th percentile of the controls' clips (0.744) is the "floor" below which a clip is further from the baseline than a numerically equivalent change ever put one.

| configuration | speedup | DINO similarity | clips below floor | vs noise | CLIP delta |
|---|---|---|---|---|---|
| perturbation controls | 1.00 | 0.876 / 0.888 | 2 / 1 of 32 | control | +0.003 / +0.007 |
| CFG truncation 0.6 | 1.22 | 0.968 | 0 | higher | -0.003 |
| WorldCache 0.02 | 1.21 | 0.949 | 0 | higher | +0.001 |
| CFG truncation 0.4 + PAB 2 | 1.43 | 0.930 | 0 | higher | -0.005 |
| first-block cache 0.05 | 1.24 | 0.865 | 4 | same | +0.006 |
| AdaCache slow30, distance_scale 4 | 1.15 | 0.959 | 1 | higher | +0.001 |
| TaylorSeer 3 | 1.35 | 0.657 | 24 | LOWER | -0.024 |
| fewer steps (20) | 1.43 | 0.633 | 27 | LOWER | -0.008 +- 0.007 |
| **WorldCache 0.04** | **1.65** | **0.834** | **6** | LOWER (slightly) | -0.005 |
| **CFG truncation 0.6 + WorldCache 0.04** | **1.98** | **0.832** | **6** | LOWER (slightly) | -0.004 |
| first-block cache 0.1 | 1.73 | 0.679 | 22 | LOWER | +0.002 |
| AdaCache slow30 (paper codebook) | 1.73 | 0.693 | 14 | LOWER | -0.017 |
| AdaCache fast30 + guard 0.10 / 0.15 | 1.55 / 1.86 | 0.768 / 0.652 | 11 / 25 | LOWER | +0.006 / +0.002 |
| AdaCache slow30, distance_scale 0.5 | 1.96 | 0.570 | 27 | LOWER | -0.004 |
| WorldCache 0.08 | 2.41 | 0.576 | 29 | LOWER | -0.035 |
| AdaCache fast30 | 2.51 | 0.454 | 31 | LOWER | -0.058 |

What this settles:

- **Up to ~1.4x nothing is lost, now on a content measure too**: CFG truncation, WorldCache 0.02, CFG + PAB and gentle AdaCache are at or above the controls' similarity (they stay closer to the baseline than a numerically equivalent change does) with no clip below the floor. First-block cache 0.05 is the same as the control (4 of 32 clips below the floor).
- **At 1.65-2x WorldCache is far ahead of everything else**: DINO similarity 0.83 with 6 of 32 clips below the floor (and 0.83 again at 1.98x with CFG truncation), against 0.68 and 22 of 32 for first-block cache at 1.73x, 0.69 and 14 of 32 for AdaCache at 1.73x, 0.57 and 27 of 32 for AdaCache at 1.96x. This is the comparison PSNR and the pixel statistics could not make cleanly; it now holds on a measure that sees content, and it matches what the contact sheets show (WorldCache keeps the baseline's scene; the others change it).
- **The drift guard helps AdaCache and does not catch WorldCache**: guarded fast30 gives 0.77 at 1.55x and 0.65 at 1.86x against 0.57-0.69 unguarded at 1.7-2x, but WorldCache is at 0.83 at 1.65-1.98x.
- **Low similarity is not the same as low quality.** DINO similarity measures sameness. `steps_20` has similarity 0.63 and 27 of 32 clips below the floor, yet its prompt match is unchanged within noise (CLIP delta -0.008 +- 0.007) and the contact sheet showed a coherent, differently composed video. Read the CLIP and consistency columns for quality: the configurations whose CLIP delta is significantly negative, and whose own temporal consistency falls, are TaylorSeer (-0.024), AdaCache slow30 (-0.017), WorldCache 0.08 (-0.035, consistency -0.031) and AdaCache fast30 (-0.058, consistency -0.039).
- **Exploratory "broken clip" rule: content changed (DINO similarity below the floor) and prompt match down (CLIP delta below -0.015).** On the 26 clips I labelled from the contact sheets it flags 6 of the 9 I judged broken or degraded and none of the 17 I judged fine, but I looked at those same clips to choose the thresholds, so treat it as a screen, not a validated detector. Across the 32 clips it flags: controls 0-2, PAB / gentle AdaCache / CFG truncation 0-1, first-block cache 0.05 0, WorldCache 0.04 2 (1.65x) and CFG + WorldCache 2 (1.98x), guarded AdaCache 4-8, first-block cache 0.1 10, AdaCache slow30 9, TaylorSeer 13, fewer steps 12, WorldCache 0.08 20, AdaCache fast30 21.
- **Limits.** General image encoders: no physics, weak on counts and fine detail, and artifacts they cannot see pass (WorldCache 0.08 on the `pan_forest` clip keeps DINO similarity 0.84 while its contact sheet shows artifacts). Both are 224-pixel models on 192x320 clips, and the prompt-match signal is small (about 0.01-0.05 in cosine) next to its clip-to-clip spread.

### Distribution-level check: Frechet distance in DINO space (the FVD stand-in)

The outline lists FVD, a distance between the *distributions* of generated and reference videos, and `visual.fvd` had been a TODO. Standard FVD embeds videos with an I3D network (not downloaded here); `worldoptbench/metrics/fvd.py` computes the same Frechet distance on a DINOv2 video descriptor (the mean and standard deviation over frames of the frame embeddings) instead, so it is a **DINO-space Frechet distance, not comparable with published FVD values**. With 32 clips per set and 768 descriptor dimensions the covariance is badly rank-deficient, so the distance is computed in the baseline set's 8-component PCA space with covariance shrinkage 0.1, calibrated against the perturbation controls at the same n, with a bootstrap over clips for the spread (`scripts/analyze_fvd.py`, results in `results/wan_fvd.md`). A smaller set of configurations than the main sweeps:

| configuration | speedup | distance | bootstrap sd | excess over the controls |
|---|---|---|---|---|
| perturbation controls | 1.00 | 0.003 / 0.002 | 0.006 / 0.003 | (floor 0.0025) |
| CFG truncation 0.6 | 1.22 | 0.001 | 0.000 | none |
| CFG truncation 0.4 + PAB 2 | 1.44 | 0.002 | 0.002 | none |
| WorldCache 0.04 | 1.66 | 0.007 | 0.005 | +0.005 |
| CFG truncation 0.6 + WorldCache 0.04 | 1.99 | 0.007 | 0.006 | +0.005 |
| first-block cache 0.1 | 1.73 | 0.022 | 0.018 | +0.019 |
| AdaCache slow30 | 1.72 | 0.039 | 0.025 | +0.037 |
| WorldCache 0.08 | 2.41 | 0.099 | 0.028 | +0.097 |
| AdaCache fast30 | 2.52 | 0.091 | 0.032 | +0.089 |

It agrees with the per-clip scores about who is fine (CFG truncation, CFG + PAB sit on the floor; WorldCache 0.04 is a small step above it) and who has left the baseline's distribution (WorldCache 0.08 and AdaCache fast30, each about three bootstrap standard deviations out). It is not a sharper instrument than the per-clip DINO similarity: for the middle configurations (first-block cache 0.1, AdaCache slow30) the bootstrap spread is as large as the effect, so it cannot separate them from WorldCache 0.04 with 32 clips. Use it as a distribution-level confirmation, not as the primary ranking.

### WorldCache (`worldcache` module), what was and wasn't reproduced

Implemented from arXiv 2603.22286's description (no code was available): motion-adaptive threshold, saliency-weighted drift, least-squares residual interpolation, threshold schedule, and optional motion-compensated warping (`warp=True`, see below). Defaults are the paper's (tau0 0.08, alpha 2, beta_s 0.12, beta_a 4, gamma_max 2), with the tau0 sweep above.

- **The literal drift rule fails.** The paper's drift compares each probe with the previous *step's* probe. Taken literally, the error from reusing stale deep features never enters the decision: it skipped 28 of 30 steps (5.0x, PSNR 10.1, skill 0). The default `drift_reference="last_computed"` compares with the last full forward (what first-block cache and TeaCache do). The literal version is available as `drift_reference="previous_step"`.
- **Ablations at tau0 0.04** (32 clips; speeds differ across ablations, so these are not speed-matched): the **threshold schedule is what creates the speedup** (off: nothing skips, 1.00x). Removing **interpolation** changed nothing measurable (1.69x, 19.7 vs 1.65x, 19.2). Removing **saliency weighting** changed nothing (identical to 3 digits). Removing **motion adaptation** gave +8% speed for -1 dB, i.e. roughly on the same trade-off curve. So on Wan the paper's four components reduce here to "a first-block-style cache with a relaxing threshold"; the other two did not show a benefit. That may be because OSI only has an effect after two *distinct* computed residuals, because beta_s is small, or because the paper's gains come from warping, which is not implemented here.
- **Motion-compensated warping does nothing here, and the reason is physical.** It is implemented as one global translation per latent frame, found by multi-scale phase correlation of the transformer's input latents and applied to the cached deep residuals (primitives recover known shifts exactly for integers and within ~0.2 px for fractions). On real Wan clips the estimated displacement between consecutive denoising steps is **0.001 latent pixels on average (max 0.003)**: the content of a video does not translate between denoising steps, it changes by being denoised, so there is no motion for a warp to compensate. Over the 32 clips `wc 0.04 + warp` gives PSNR 19.1 vs 19.2, style deviation 0.131 vs 0.126, and is slower (1.50x vs 1.65x: ~0.8 s of FFTs and shifts per clip, versus the paper's 'under 3%'). Off by default. The paper's version may estimate something richer than a global translation (a dense or local motion field), but a physical argument says consecutive-step latents carry no spatial motion, so I would not expect a different result; I did not test a dense variant.
- **The paper's headline (2.3x at 99.4% quality) is not reproduced**, and cannot be with this metric: the paper reports PAI-Bench quality on Cosmos-Predict2.5; here the same speedup (2.40x) has PSNR 13.4 against the unoptimized video. Those are different questions. What is reproduced is the qualitative claim that an adaptive threshold beats a fixed first-block threshold at equal speed.


### AdaCache (`adacache` module)

Implemented from arXiv 2411.02397's description (no code was available): per-block residual caching of attention, cross-attention and MLP outputs with one global cache rate, chosen at each full computation from a codebook keyed by the per-step change between the last two computed steps; optional motion regularization. Presets are the paper's codebooks (`slow30`, `fast30`, `fast100`); `distance_scale` rescales the measured distance, which is model-dependent. Same 32 clips, same noise control (style deviation 0.081); full table in `results/wan_comparison.md`.

| configuration | speedup | PSNR | style deviation | vs noise |
|---|---|---|---|---|
| slow30, distance_scale 4 | 1.15 | 34.7 | 0.051 | within |
| slow30, distance_scale 2 | 1.47 | 21.2 | 0.318 | beyond |
| slow30 (paper's codebook) | 1.73 | 18.5 | 0.318 | beyond |
| fast30, distance_scale 4 | 1.77 | 19.1 | 0.380 | beyond |
| slow30, distance_scale 0.5 / 0.25 | 1.96 / 1.98 | 13.6 / 13.1 | 0.392 / 0.464 | beyond |
| fast30, distance_scale 2 | 2.14 | 14.1 | 0.647 | beyond |
| fast30 (paper's codebook) | 2.51 | 11.1 | 0.775 | beyond |
| fast30 + motion regularization | 2.53 | 11.9 | 0.748 | beyond |

- **On Wan the paper's codebooks are too aggressive**: Wan's per-step distances are 0.02-0.2 on my measure, so the first thresholds put most steps at the longest rate. Scaling the distance up gives gentler points; scaling it down saturates at the codebook's longest rate (slow30 stops near 2x).
- **At equal speed AdaCache is clearly worse than WorldCache and worse than the simple options**: at ~1.7x slow30 has deviation 0.318 against WorldCache's 0.126 (and first-block cache's 0.296); at ~2x, 0.39-0.46 against 0.125 for CFG truncation + WorldCache; at ~2.5x, 0.775 against 0.429 for WorldCache 0.08. Below 1.5x, CFG truncation + PAB (1.44x, 0.057, within noise) beats it too. Its gentlest point (1.15x, PSNR 34.7) is the highest PSNR measured but is dominated by CFG truncation 0.6 (1.22x, deviation 0.040).
- **Motion regularization did not help measurably**: fast30 deviation 0.775 vs 0.748 with it, and 0.647 vs 0.544 at scale 2 (each about one standard error apart).
- This is the opposite of the paper's ordering (AdaCache over PAB at matched latency, on Open-Sora, Open-Sora-Plan and Latte). Possible reasons, none tested: my reading of how the paper combines per-layer distances and handles classifier-free guidance (separate schedules per CFG branch here), a short 30-step schedule on a 1.3B model at low resolution, or its codebooks being tuned to other models' activation scales. Treat it as "this implementation on this model", not a refutation of the paper.

### Drift guard: a portable safety net under a cache (`AdaCacheModule(guard=...)`)

The outline's drift-correction idea (section 3.5) was flagged as overlapping WorldCache's own detect-and-recompute loop, and its candidate rescopings were left undecided. Of those, the one testable with what exists here is "drift correction as a layer any cache can sit under". Quantization-induced drift correction is moot (quantization was slower on Dreamer), the autoregressive variant has no model, and token-level selective recompute through attention was not attempted. What was built: a guard for AdaCache (not in the paper), the cache that most lacks its own drift signal, since it can only see change at the steps it recomputes and runs blind between them.

With `guard=g`, the first `guard_blocks` (default 1) blocks run as a probe on every step. Their relative change since the last full computation (mean absolute difference over mean absolute value, averaged over probe modules) is compared with `g` once the probe is done; above it, the scheduled reuse is vetoed, the step recomputes everything, and the codebook restarts the schedule as for any computed step. The probe costs about 6% (one block of 30), included in the speeds below. Same 32 clips and noise control (style deviation 0.081).

| configuration | speedup | PSNR | style deviation | vs noise |
|---|---|---|---|---|
| fast30 + guard 0.05 | 1.18 | 24.8 | 0.053 | below |
| fast30 + guard 0.10 | 1.54 | 17.7 | 0.197 | beyond |
| slow30 + guard 0.15 | 1.58 | 19.2 | 0.294 | beyond |
| fast30 + guard 0.15 | 1.85 | 15.0 | 0.314 | beyond |
| fast30 + guard 0.25 | 2.22 | 12.8 | 0.522 | beyond |
| CFG truncation 0.6 + fast30 + guard 0.15 | 2.21 | 15.0 | 0.318 | beyond |

- **The guard helps AdaCache, moderately.** At matched speed the guarded configurations sit lower than the unguarded ones: 1.54x at 0.197 against 1.47x at 0.318 (unguarded slow30, scale 2), 1.85x at 0.314 against 1.77x at 0.380, 2.22x at 0.522 against 2.14x at 0.647. Its threshold is a real speed/quality dial (1.18x within noise up to 2.2x at 0.52).
- **It does not close the gap to WorldCache**: 1.65x at 0.126 for WorldCache against 1.54x at 0.197 and 1.85x at 0.314 for the guarded AdaCache. That is unsurprising, since WorldCache is already built around a per-step probe with accumulated drift plus residual extrapolation; the guard gives AdaCache the first of those but not the second.
- **CFG truncation 0.6 + guarded AdaCache (2.21x at 0.318) is on the Pareto front** but not clearly ahead of the line between the WorldCache combinations (1.99x at 0.125 and 2.40x at 0.429).
- **Scope and limits.** Built and measured on AdaCache only; not attached to PAB, first-block cache or the diffusers caches, so "portable" is shown by construction (it needs only "reuse is scheduled"), not demonstrated across caches. No token-level selective recompute. The outline's targets (recover 60-80% of the physics gap while keeping WorldCache's speedup) are not testable here: there is no physics score for Wan, and the guarded AdaCache does not keep WorldCache's speedup at WorldCache's deviation.

### Combined comparison

`python scripts/compare_diffusion.py results/wan_all --out results/wan_comparison.md` merges the sweeps into one table with the noise control and the Pareto front. On speed and style deviation, the front is: CFG truncation 0.6 (1.22x), CFG truncation 0.4 + PAB 2 (1.44x), CFG truncation 0.6 + WorldCache 0.04 (1.99x), CFG truncation 0.6 + guarded AdaCache (2.21x), WorldCache 0.08 (2.40x), and, by speed alone, AdaCache fast30 with motion regularization (2.53x). The only AdaCache configurations on it are the guarded one at 2.21x and that fastest point.

### More optimizations from the catalog, and MagCache (2026-10-04)

A second round from `OPTIMIZATION_CATALOG.md`, all on the same 32 clips, with DINO similarity to the unoptimized video (noise controls 0.876 / 0.888, floor 0.744). Results in `results/wan_more*/`, table in `results/wan_perceptual_all.md`, plot `results/wan_pareto.png`.

| configuration | speedup | DINO similarity | clips below floor | vs noise | note |
|---|---|---|---|---|---|
| **MagCache 0.06** | 1.49 | 0.885 | 2 | same | calibrated ratios |
| **MagCache 0.12** | 1.80 | 0.861 | 4 | same | |
| **MagCache 0.24** | 2.03 | 0.855 | 4 | same | thresholds above this add nothing (bounded by `max_skip_steps` 3 and `retention_ratio` 0.2) |
| MagCache 0.24, `max_skip_steps` 5 | 2.15 | 0.833 | 5 | lower | |
| MagCache 0.24, `retention_ratio` 0.1 | 2.30 | 0.646 | 23 | lower | **collapses**: the early steps must be computed |
| **MagCache 0.24 + CFG truncation 0.6 + KV cache + bf16 VAE** | **2.54** | **0.856** | 3 | same | prompt match +0.003 |
| MagCache 0.5 + CFG truncation 0.6 + KV cache + bf16 VAE | 2.63 | 0.847 | 4 | same | |
| KV cache + bf16 VAE + CFG truncation 0.4 + PAB 2 | 1.54 | 0.930 | 0 | higher | the default stack |
| cross-attention KV cache | 1.02 | 1.000 | 0 | higher | exact, identical output |
| bf16 VAE | 1.04 | 0.999 | 0 | higher | peak memory 3.58 GB against 4.40 GB |
| VAE tiling | 0.98 | 1.000 | 0 | higher | memory only (4.15 GB) |
| TF32 / cuDNN benchmark | 1.00 | 1.000 | 0 | higher | identical output, no gain |
| scheduler, 20 steps: UniPC / DPM++ / Euler / order 3 | 1.43 | 0.633 / 0.644 / 0.639 / 0.643 | 24-27 | lower | the solver does not matter |
| scheduler, 20 steps, flow shift 5 / 8 / 1.5 | 1.43 | 0.764 / 0.650 / 0.554 | 10 / 24 / 28 | lower | shift 5 is the best of the fewer-steps variants |
| unconditional-pass reuse, period 2 / 3 | 1.29 / 1.43 | 0.358 / 0.222 | 31 / 32 | lower | **harmful** (prompt match -0.08 / -0.12) |

What it settles:

- **MagCache is the best cache on Wan by a wide margin**: at 2.0x its DINO similarity (0.855) is statistically the same as a numerically equivalent change (0.876), where WorldCache at 1.98x is 0.832 (below the controls), first-block cache at 1.73x is 0.68 and AdaCache at 1.7-2x is 0.57-0.69. With CFG truncation and the lossless extras it reaches **2.54x at 0.856**, 3 of 32 clips below the floor, prompt match unchanged. It needs per-setup magnitude ratios (measured once, `calibrate_mag_ratios`; the Wan ones are bundled and `WanVideo.mag_ratios` returns them only for the calibrated size, step count and sampler).
- **The contact sheets agree** (`results/contact_sheets/*_magcache.png`, six prompts, two viewed): every MagCache variant keeps the baseline's scene, layout and motion, where WorldCache 0.04 introduces visible shape artifacts on the same clips; the retention-0.1 variant is visibly broken on `ball_ramp`. Pixel statistics show no sharpness loss on average (ratio 0.97-0.98 against 0.94 for the noise control) but a small measurable style shift (deviation 0.108-0.129 against 0.080), and a mild softening of fine detail is visible on some clips.
- **A free tier exists**: KV cache + bf16 VAE on top of CFG truncation 0.4 + PAB 2 gives 1.54x at the same fidelity as 1.43x without them (DINO 0.930, 0 of 32 clips below the floor). `recommended_config(model)` returns it; `recommended_config(model, aggressive=True)` returns the MagCache stack when the model has ratios for its setup. Verified end to end on the real model: 1.51x and 2.54x.
- **The unconditional pass cannot be reused**: its prediction changes enough between steps that stale ones break guidance; use `cfg_truncation` to thin it.
- **The solver does not matter, the timestep shift does** (optimum near 5 at 20 steps), but fewer steps stays far below the caches at the same speed.
- **Quantization is slower on Wan**, measured on one clip per scheme (timings only): int8 weight-only 0.85x, fp8 weight-only 0.73x, fp8 dynamic 0.80x, int8 dynamic 0.23x, saving about 1.1 GB of peak memory; int4 needs the `mslk` package. Without `torch.compile` (no Triton or MSVC here) torchao's fused kernels are unavailable.

Not built from the catalog, with reasons: `torch.compile` (no toolchain), SageAttention, xFormers and FlashAttention-3 (package-gated), token-wise and region caches, sparse attention, token merging (substantial new algorithms), speculative decoding, step distillation and every K-series (multi-GPU) and TRAIN item.

### Library gotchas found while doing this (all fixed or documented)

- diffusers' `HookRegistry._get_child_registries` caches the list of child registries on the first `cache_context(...)` and never invalidates it. A cache applied (or removed) after the transformer has already run once never receives its context ("No context is set"). Earlier sweeps avoided this only because each process applied its cache before the first generation. All cache modules now call `refresh_hook_registries` after applying or removing hooks.
- The pipeline does reset stateful hooks between videos (`maybe_free_model_hooks` at the end of `__call__`); I briefly believed it did not, and that the earlier TaylorSeer rows were contaminated. They were not.
- Per-process peak memory: the first run of a sweep also holds the reference generator, which is why one baseline row shows 7.5 GB and the others 4.4 GB.
