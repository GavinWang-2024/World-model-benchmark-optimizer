# Optimization catalog

Every optimization we could plausibly add to the WorldOptBench stack: **126 entries** across 14 categories. Generated from one table, so the counts below always match the rows. Written 2026-10-03.

Evidence labels: entries marked **Measured** were benchmarked on the trained Dreamer walker (results in `DREAMER_SETUP.md`). Everything else is a plan, not a result. Details on 2026 papers came from search-result summaries, not from reading the papers in full — read the paper before relying on a number.

## Status summary

| Status | Count | Meaning |
|---|---|---|
| **BUILT** | 42 | Built and measured on hardware |
| **EXP** | 6 | Built and unit-tested; not yet measured on hardware (maturity: experimental) |
| **NOW** | 18 | Buildable now (no new access) |
| **DIFF** | 25 | Needs a diffusion/AR transformer world model |
| **MULTI** | 10 | Needs multiple GPUs |
| **TRAIN** | 12 | Needs training or fine-tuning |
| **TOOL** | 5 | Blocked by toolchain / dependency here |
| **RES** | 8 | Open research or unclear payoff |
| **Total** | 126 | |

**Applies to:** D = diffusion, R = recurrent latent model (Dreamer), AR = transformer autoregressive, J = JEPA, all = any architecture. **Effort:** S = hours, M = a day or two, L = a week or more (or needs training). **Physics risk:** how likely the change is to move the physics score; `n/a` for items that exist to improve it.

## How the library is used

- **See what applies:** `library_report(model)` lists every registered module with whether it can be applied to *that* model on *this* machine, the reason if not, its maturity, and a one-line summary.
- **Hand it everything:** `OptimizationStack(model, available_modules())` is safe. A module whose requirements aren't met (wrong architecture, missing hook, no CUDA, missing package) is skipped and recorded in `.skipped` with the reason.
- **Let it choose:** `autotune(candidates, evaluate)` starts with no optimization and adds whichever module most improves PAES, stopping when nothing helps by more than `min_gain` (default 10%, because single eager runs vary +-34%) and rejecting any stack that costs more than `max_physics_drop` of physics. `evaluate_with_runner(model_factory, ...)` builds the real evaluator.
- **Maturity:** `measured` modules have a recorded effect in `DREAMER_SETUP.md`. `experimental` modules are implemented and unit-tested but have not been swept on hardware: don't assume they help. New modules start experimental.

## Hooks the stack needs

Modules plug in through small opt-in extension points on the model wrapper (the model declares support with a Protocol; the module refuses models that don't). Three exist today; the rest are the real work behind most of the unbuilt items.

| Hook | State | Purpose |
|---|---|---|
| `EXEC` | `tensor_executor` — exists | how the hot tensor function runs (CUDA graphs live here) |
| `AUTOCAST` | `autocast_dtype` — exists | mixed precision |
| `TORCH` | `torch_module()` — exists | access to the `nn.Module` to modify in place |
| `STEPHOOK` | per-step callback inside the rollout — **not built** | needed for early stop, re-anchoring, gating, sparse decode, drift correction; conflicts with whole-rollout CUDA graphs unless chunked (A6) |
| `DENOISE` | diffusion denoise-loop hook — **not built** | before/after/skip a denoising step or block; needed by every diffusion item |
| `BATCH` | `generate_batch` — **not built** | run several rollouts per call; needs a throughput metric |
| `KV` | KV-cache interface — **not built** | transformer autoregressive models only |
| `SERVER` | worldserve scheduler — **not built** | request queue, batching, caching |
| `TRAIN` | training pipeline — **not built** | anything that changes weights |
| `none` | no hook needed |  |

## A. Execution and compilation

Make the same math run with less overhead. No change to outputs unless noted.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| A1 | CUDA graphs | Replay the whole rollout (context + imagination + decode) as one GPU launch. **Measured on Dreamer: ~9x vs eager (eager baseline is noisy, +-34%); 21.7 ms mean, output essentially identical.** | all | `EXEC` | none | S | **BUILT** |
| A2 | torch.compile (inductor) | Fuse kernels and cut launch overhead automatically. Needs Triton plus a C compiler; this machine has neither (Windows, no MSVC). Would work on Linux/WSL. | all | `TORCH` | none | S | **TOOL** |
| A3 | torch.compile reduce-overhead / max-autotune modes | Compile with built-in CUDA graphs and kernel autotuning. Same toolchain block as A2. | all | `TORCH` | none | S | **TOOL** |
| A4 | TensorRT engine for the RSSM step | ONNX-export one imagination step, build a TensorRT engine, run it per step inside the CUDA graph. **Measured: 13.2 ms mean, 1.33x faster than the pure-PyTorch step in graphs and 1.64x faster than the repo loop in graphs; matches its PyTorch reference to 2.5e-6 in skill.** Uses Gumbel-max sampling (A14). Needs `tensorrt-cu12` + `onnx`; never `torch-tensorrt` next to an older torch. | R | `EXEC` | low | M | **BUILT** |
| A5 | Hand-fused RSSM-cell kernel | One custom kernel for the whole step. TensorRT already gets the step to ~28 us, so this only matters if that is not enough. No nvcc or host compiler here; CuPy RawKernel (NVRTC) could compile it without one. | R | `TORCH` | none | L | **TOOL** |
| A6 | Chunked graphs (capture k steps, replay ceil(N/k) times) | `chunked_rollout` module (`models/chunked.py`). **Measured (Dreamer 515k, TensorRT fp16 + graphs): output matches the whole-rollout graph to one grey level; NOT a speedup: ~0.2 ms per chunk boundary (+3-5% at 50-100 steps per chunk on long rollouts, +25-45% at 10-20), large chunks waste work on short horizons, three graphs serve every horizon but held memory was higher (215-485 MB vs 126 MB), not lower.** Keep whole-rollout graphs when no hooks are needed; chunking is the base for early stop / re-anchoring / streaming. Needs gumbel_sampling or tensorrt (explicit noise). | R | `EXEC` | none | M | **BUILT** |
| A7 | Pre-allocated pinned input buffers + async host-to-device copies | Measured: host-side overhead (copies, tensor creation, RNG fork, readback) is only ~0.4 ms of a 4-5 ms rollout, so there is little left to remove. (An earlier note claimed a ~20 ms floor; that was a pre-fusion artifact.) | all | `EXEC` | none | S | **RES** |
| A8 | Multi-stream overlap | Decode frames on a side stream while the next steps run; overlap copies with compute. | all | `EXEC` | none | M | **NOW** |
| A9 | channels_last memory format for conv encoder/decoder | `channels_last` module. **Measured: no gain** (23.6 vs 21.0 ms with graphs, inside the 13% noise); identical output. | all | `TORCH` | none | S | **BUILT** |
| A10 | cuDNN benchmark / autotune | `cudnn_benchmark` module. **Measured: no effect** on the Dreamer model (20.9 vs 21.0 ms with graphs), nor on Wan2.1-1.3B (32 clips: identical output, 1.00x). | all | `TORCH` | none | S | **BUILT** |
| A11 | Graph memory-pool sharing across shapes | `CudaGraphsModule(share_pool=True)`. **Measured: saves 172 MB held (672 -> 500 MB, -26%) with identical output**; the latency effect is unclear (27.5 ms on the random-action set, roughly equal to plain graphs on the policy set) — re-measure before relying on it. | all | `EXEC` | none | S | **BUILT** |
| A12 | Pre-capture graphs at startup (warm pool) | `DreamerWorldModel.warmup(horizons, batch_size)` runs zero-valued rollouts; `WorldModelServer(..., warmup=[requests])` now runs sample requests inside `start()`, before accepting traffic, so graph capture and TensorRT builds do not land in the first real request. The benchmark runner's own warm-up already does this for benchmarks. | all | `EXEC` | none | S | **EXP** |
| A13 | Disable torch.distributions argument validation | Each distribution's parameter check costs a GPU->CPU sync. **Measured in a controlled A/B: 1.43x on the eager path** (431 -> 302 ms at 20 s); the 3-repeat sweep could not confirm it (eager noise 31%). No effect with graphs, which already disable it during capture. | all | `none` | none | S | **BUILT** |
| A14 | TensorRT reduced-precision engines (fp16) | `TensorRTModule(precision="fp16")`. **Measured: 9.9 ms vs 13.1 ms for fp32 (1.32x faster), faster at every horizon, physics matching** (mean skill diff 0.007 vs ~0.05 sampling noise; PSNR within 0.2 dB; skill 0.251 vs 0.250 on held-out walking). The fastest configuration so far. int8 not attempted. | R | `EXEC` | low | M | **BUILT** |
| A15 | TensorRT for the decoder | `tensorrt_decoder` module. **Measured: output identical to the PyTorch decoder (zero difference), ~8-10% faster end to end** (12.0 vs 13.1 ms; 23.8 vs 26.2 ms at 20 s). Its reported memory drop is TensorRT workspace that PyTorch doesn't count, not a saving. Encoder not done. | R | `EXEC` | low | M | **BUILT** |
| A16 | Whole-rollout TensorRT engine (unrolled N steps) | One engine for all N steps instead of one enqueue per step. Huge graph and long build; unclear payoff over per-step engines in a CUDA graph. | R | `EXEC` | low | L | **RES** |

## B. Numerics and quantization

Cheaper arithmetic or smaller weights. Changes numerics, so physics must be re-measured.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| B1 | TF32 matmuls | TF32 tensor-core mode for float32 matmuls. **Measured: no demonstrated effect (inside run-to-run noise) on Dreamer; on Wan2.1-1.3B (32 clips) the output is identical and the speed unchanged (1.00x), since the transformer already runs in bf16.** An earlier ~4% claim on Dreamer was not reproduced. | all | `none` | low | S | **BUILT** |
| B2 | bf16 autocast | Mixed precision in bfloat16. **Measured on Dreamer: slower with graphs (31.5 vs 21.7 ms), skill -0.004, ~100 MB less memory.** | all | `AUTOCAST` | low | S | **BUILT** |
| B3 | fp16 autocast | Mixed precision in float16. **Measured: slower with graphs (27.0 vs 21.7 ms), skill +0.003.** | all | `AUTOCAST` | low | S | **BUILT** |
| B4 | INT8 weight-only (TorchAO) | 8-bit weights, dequantized in the matmul. **Measured: no gain (23.1 vs 21.7 ms with graphs; eager is inside the noise); skill -0.002; +58 MB held.** | all | `TORCH` | low | S | **BUILT** |
| B5 | INT8 dynamic (activations + weights) | **Measured earlier (single run): far slower (~0.05x) on this model.** | all | `TORCH` | low | S | **BUILT** |
| B6 | FP8 weight-only and dynamic (TorchAO) | TorchAO fp8 via the `quantization` module. **Measured on Dreamer (single runs): 0.87x and 0.37x. On Wan2.1-1.3B (one clip per scheme, timings only): fp8 weight-only 0.73x, fp8 dynamic 0.80x, both SLOWER than bf16 without `torch.compile`; output at the noise floor.** Needs compute capability 8.9+. | all | `TORCH` | low | S | **BUILT** |
| B7 | INT4 weight-only (TorchAO) | INT4 weight-only (TorchAO). Fails here: TorchAO wants the extra `mslk >= 1.0.0` package, which is not installed. On Wan the other schemes were all slower than bf16 (int8 weight-only 0.85x, int8 dynamic 0.23x; peak memory about 1.1 GB lower), so int4 is unlikely to help speed either. | all | `TORCH` | med | S | **TOOL** |
| B8 | NVFP4 / MXFP4 (Blackwell) | 4-bit float on Blackwell tensor cores (this GPU is sm_120). TorchAO support is prototype-stage; unclear it works on a laptop part. | all | `TORCH` | high | M | **RES** |
| B9 | SmoothQuant-style W8A8 with calibration | Calibrate on real rollouts to tame activation outliers, then quantize both weights and activations. | all | `TORCH` | med | M | **NOW** |
| B10 | Sensitivity-guided mixed precision | Measure each layer's effect on physics and keep only the sensitive ones in high precision. Directly attacks quantization drift. | all | `TORCH` | low | M | **NOW** |
| B11 | Weight-cast half precision (model.half()) instead of autocast | Store weights in half precision; skips per-call casts. Needs a dtype-aware core; RSSM distribution sampling may be fragile. | all | `TORCH` | med | M | **NOW** |
| B12 | Quantized decoder/encoder convolutions | TorchAO's Linear-only configs skip the conv layers. Intx weight-only supports conv per its docs. | all | `TORCH` | med | M | **NOW** |
| B13 | SVDQuant (4-bit with low-rank outlier branch) | Aggressive 4-bit for DiTs. | D | `DENOISE` | med | L | **DIFF** |
| B14 | SageAttention (8-bit / 4-bit attention kernels) | Quantized attention kernels for transformers (Jintao Zhang et al.). | D, AR | `DENOISE` | low | M | **DIFF** |
| B15 | Quantized KV cache (INT8 / FP8) | Shrink the KV cache of a transformer world model; helps long horizons. | AR | `KV` | med | M | **DIFF** |
| B16 | Quantization-aware training | Train with fake-quant so 4-bit is usable. Needs GPU hours. | all | `TRAIN` | low | L | **TRAIN** |

## C. Recurrent / autoregressive algorithms

Do less work per generated frame. Mostly Dreamer-specific.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| C1 | Lean scan | Replace the repo's growing-`torch.cat` loop with collect-and-stack. **Measured: no demonstrated effect (21.1 vs 21.7 ms with graphs, inside the noise); identical output.** | R | `none` | none | S | **BUILT** |
| C2 | Batched rollouts | `DreamerWorldModel.generate_batch(requests)` runs B same-shaped rollouts in one batched forward (the decode output now keeps its batch dimension). Shape validation unit-tested; throughput never measured. One seed per batch, so outputs differ from single calls (statistically equivalent). The TensorRT step engine is batch-1 only. | R | `BATCH` | none | M | **EXP** |
| C3 | Sparse decoding | `sparse_decode` module. **Measured: 9 / 13 / 15% faster at strides 2 / 4 / 8, but real quality loss on held-out walking** (skill -0.014 / -0.047 / -0.079 vs the same-noise reference). On long random-action horizons pixel scores *rose* (blur is rewarded by a pixel metric) — a metric artifact. Stride 2 is a mild trade; not a default. | R | `STEPHOOK` | med | M | **BUILT** |
| C4 | Latent-only rollouts | `DreamerWorldModel.generate_latents(...)`: the imagined rollout as latent features, skipping the decoder, for consumers that never look at pixels (planners, policy evaluators). **Measured (Dreamer 515k, median of 15): 17-18% less time with CUDA graphs at every horizon; with TensorRT fp16 + CUDA graphs 23% at 2 s up to 42% at 20 s (the decoder is a larger share once the dynamics step is fast).** The eager path is too noisy to read. Same seeds and knobs as `generate()`. | R | `STEPHOOK` | none | S | **BUILT** |
| C5 | Context-latent cache | Cache the posterior state for repeated contexts so the context phase is skipped. | R | `STEPHOOK` | none | S | **NOW** |
| C6 | Uncertainty-gated early stop | End a rollout when prior/posterior divergence or latent entropy says it has stopped being trustworthy. | R | `STEPHOOK` | low | M | **NOW** |
| C7 | Speculative rollouts with a small draft model | A small world model drafts latent steps, the large one verifies. Needs two trained sizes and an acceptance rule on latents. | R | `TRAIN` | med | L | **TRAIN** |
| C8 | Action-chunked / temporally abstract rollouts | Take larger time steps by repeating actions, so fewer model steps per second of simulated time. | R | `none` | high | M | **RES** |
| C9 | Parallel-scan (linearized) recurrence | Replace the sequential GRU with something parallelizable over time. Changes the model; research. | R | `TRAIN` | high | L | **RES** |
| C10 | Smaller decoder / lower-res decode + upsample | Cheaper image head. Needs retraining. | R | `TRAIN` | med | L | **TRAIN** |
| C11 | KV caching for transformer autoregressive world models | Standard LLM-style KV cache for AR video/token models (e.g. Genie-style). No such model in v1. | AR | `KV` | none | M | **DIFF** |
| C12 | Speculative decoding for autoregressive world models | LLM-style speculative decoding applied to AR world models (outline section 4). | AR | `KV` | low | L | **DIFF** |
| C13 | Gumbel-max sampling step (pure PyTorch) | Rewrite the RSSM step with exportable ops and Gumbel-max sampling: same distribution as `torch.multinomial` (180 rollouts each: pooled skill 0.274 vs 0.276, z = -0.36), different random stream. **Measured: 17.6 ms in graphs vs 21.7 ms for the repo loop, and ~1.6x faster eager.** Also the exact reference TensorRT is checked against. | R | `none` | low | M | **BUILT** |

## D. Diffusion: fewer or cheaper denoising steps

All need a diffusion model (Cosmos or an open DiT); none apply to Dreamer.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| D1 | Better ODE solvers (DPM-Solver++, UniPC) | `scheduler` module. **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps; DINOv2 similarity to the unoptimized video (noise controls 0.88)): at 20 steps (1.43x) the solver makes no difference: default UniPC 0.633, DPM-Solver++ 0.644, Euler 0.639, UniPC order 3 0.643.** Wan's default UniPC order 2 is already a strong solver. See D2 for the one thing that did help. | D | `DENOISE` | low | S | **BUILT** |
| D2 | Timestep schedule / shift tuning | `scheduler` module with `flow_shift`. **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps; DINOv2 similarity to the unoptimized video (noise controls 0.88)): at 20 steps shift 5 gives DINO similarity 0.764 against 0.633 for the default shift 3 (prompt match unchanged), shift 8 gives 0.650, shift 1.5 gives 0.554; at 15 steps shift 5 gives 0.688.** The best fewer-steps variant, but still far below CFG truncation + PAB at the same speed (0.930). | D | `DENOISE` | low | S | **BUILT** |
| D3 | CFG truncation | `cfg_truncation` module. **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps, baseline 12.5 s; PSNR = agreement with the unoptimized video, mean +- sem over clips.): after 60% of steps 1.21x at PSNR 30.8 +- 0.6 / SSIM 0.965; after 40%, 1.36x at PSNR 25.9 +- 0.6.** Best fidelity per unit of speed at up to ~1.4x. Composes with caches: with WorldCache 0.04 it reaches 1.98x at the same PSNR as the cache alone (1.65x). | D | `DENOISE` | low | S | **BUILT** |
| D4 | Cache or skip the unconditional CFG branch | `uncond_reuse` module. **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps; DINOv2 similarity to the unoptimized video (noise controls 0.88)): HARMFUL. Reusing the unconditional pass every 2nd step gives 1.29x but DINO similarity 0.36 (31 of 32 clips beyond the noise floor, prompt match -0.08); every 3rd gives 1.43x at 0.22.** The unconditional prediction does not change slowly enough to reuse; `cfg_truncation` (D3) is the safe way to thin it. | D | `DENOISE` | med | M | **BUILT** |
| D5 | Adaptive step count per sample | Fewer steps for easy content, more for hard. | D | `DENOISE` | med | M | **DIFF** |
| D6 | Few-step then refine | Draft quickly in few steps, refine selected regions or frames. | D | `DENOISE` | med | M | **DIFF** |
| D7 | Step distillation (DMD / consistency / AccVideo) | Distill to few steps. The outline cites AccVideo at 8.5x. XPENG also reports few-step distillation for world models. Needs training. | D | `TRAIN` | med | L | **TRAIN** |
| D8 | Guidance distillation | Fold CFG into the model so one pass does the work of two. | D | `TRAIN` | low | L | **TRAIN** |

## E. Diffusion: caching and reuse

Skip recomputation by reusing features across steps or regions. The outline's core diffusion work.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| E1 | WorldCache | `worldcache` module, implemented from the paper (arXiv 2603.22286) as hooks: motion-adaptive threshold, saliency-weighted drift, least-squares interpolation, threshold schedule, and optional motion-compensated warping (global per-frame phase correlation; **measured as a no-op**: displacement between consecutive-step latents ~0.001 latent px, PSNR 19.1 vs 19.2, ~11% slower, off by default). **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps, baseline 12.5 s; PSNR = agreement with the unoptimized video, mean +- sem over clips.): tau0 0.02 -> 1.21x, PSNR 27.0; 0.04 -> 1.65x, PSNR 19.2; 0.08 -> 2.40x, PSNR 13.4.** On reference-free video statistics (blind.py) against a numerical-perturbation control: tau0 0.02 is below the control (no measurable loss), 0.04 a small shift (flicker ~+17%), 0.08 a large one (motion +65%, flicker x2.2); at ~1.7x it is clearly better than first_block_cache (style deviation 0.126 vs 0.296). It loses to CFG truncation up to ~1.4x, and CFG truncation 0.6 + WorldCache 0.04 gives 1.99x at the same shift. Ablations at tau0 0.04: the threshold schedule is what makes it skip at all (off: 1.00x); interpolation and saliency weighting showed no measurable benefit; removing motion adaptation gave +8% speed for -1 dB. The paper's literal previous-step drift skipped 28 of 30 steps and destroyed the video (5.0x, PSNR 10.1), so the default compares against the last full forward. **Does not reproduce the paper's 2.3x at 99.4% quality**: different metric (PAI-Bench there, agreement with the baseline video here) and different model size. WorldCache + CFG truncation 0.6 at tau0 0.04: 1.98x, PSNR 19.4. On learned scores (DINOv2 similarity to the baseline video, controls 0.88): 0.83 at 1.65x and 1.98x with 6 of 32 clips below the noise floor, against 0.68 / 22 of 32 for first_block_cache at 1.73x and 0.57-0.69 for AdaCache at 1.7-2x. | D | `DENOISE` | low | M | **BUILT** |
| E2 | AdaCache | `adacache` module, implemented from the paper (arXiv 2411.02397): per-block residual caching with a global cache rate from a codebook. **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps, baseline 12.5 s; PSNR = agreement with the unoptimized video, mean +- sem over clips.), reference-free style deviation vs the baseline video (noise control 0.081): the paper's codebooks are too aggressive for Wan (slow30 1.73x, 0.318; fast30 2.51x, 0.775); distance_scale 4 on slow30 gives 1.15x at 0.051 (within noise).** At equal speed clearly worse than WorldCache (1.65x: 0.126) and, below 1.5x, worse than CFG truncation + PAB; motion regularization showed no measurable benefit. Opposite of the paper's ordering; see DESIGN_DIFFUSION.md for caveats. | D | `DENOISE` | low | M | **BUILT** |
| E3 | TeaCache | `first_block_cache` module (diffusers FirstBlockCache, TeaCache-style). **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps, baseline 12.5 s; PSNR = agreement with the unoptimized video, mean +- sem over clips.): threshold 0.05 -> 1.23x, PSNR 21.5 +- 0.8; 0.1 -> 1.72x, PSNR 14.6 +- 0.8** (0.2 -> 2.62x, PSNR ~10 on 4 clips). 0.05 is indistinguishable from a numerical perturbation; 0.1 shifts the video's statistics clearly (sharpness -23%, flicker +27%). WorldCache is better at ~1.7x and CFG truncation below ~1.4x. | D | `DENOISE` | low | M | **BUILT** |
| E4 | FasterCache | `fastercache` module. **Unavailable on Wan:** it needs the conditional and unconditional passes batched in one call and Wan's pipeline runs them separately, so the module reports itself incompatible instead of crashing. Unmeasured. | D | `DENOISE` | low | M | **BUILT** |
| E5 | Pyramid Attention Broadcast (PAB) | `pab` module (diffusers). **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps, baseline 12.5 s; PSNR = agreement with the unoptimized video, mean +- sem over clips.): skip range 2 -> 1.07x, PSNR 26.6 +- 0.6.** Small speedup, good fidelity. Stacked on CFG truncation it adds about 5% at no measurable fidelity cost (cfg 0.4 + pab 2: 1.43x, PSNR 26.3 vs 1.36x, PSNR 25.9 for cfg 0.4 alone). | D | `DENOISE` | low | M | **BUILT** |
| E6 | TaylorSeer / feature forecasting | `taylorseer` module. **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps, baseline 12.5 s; PSNR = agreement with the unoptimized video, mean +- sem over clips.): interval 3 -> 1.34x, PSNR 14.5 +- 0.6** (interval 5 on 4 clips: 1.43x, PSNR ~11). Dominated by CFG truncation 0.4 (1.36x, PSNR 25.9). An earlier no-effect result was a bug (diffusers default patterns match nothing on Wan); fixed and tested. | D | `DENOISE` | low | M | **BUILT** |
| E7 | Block-wise cache / layer skipping (DeepCache-style, Delta-DiT) | `layer_skip` module (fixed middle blocks, diffusers). **Measured (Wan2.1-1.3B, 4 prompts x 1 seed, 33 frames at 192x320, 30 steps, baseline 12.4 s; fidelity = agreement with the unoptimized video.): 2 blocks -> 1.06x, 4 blocks -> 1.14x, PSNR ~10.** Wrecks quality for little gain; not worth it on Wan. Not the DeepCache/Delta-DiT cache-the-expensive-blocks variant, which remains unbuilt. | D | `DENOISE` | low | M | **BUILT** |
| E8 | Token-wise cache (ToCa) | Choose which tokens to recompute per step. | D | `DENOISE` | med | M | **DIFF** |
| E9 | X-Cache-style region reuse | XPENG's training-free world-model accelerator (reported 2.7x): reuses image regions the physical-world continuity says haven't changed. | D | `DENOISE` | med | M | **DIFF** |
| E10 | Text-embedding and cross-attention K/V cache | `cross_attn_kv_cache` module. **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps; DINOv2 similarity to the unoptimized video (noise controls 0.88)): exact (identical output, DINO similarity 1.000) but only 1.02x** (the text keys/values are about 2% of the compute), at +0.2 GB of cache. Free, small; part of the default diffusion stack. | D | `DENOISE` | none | S | **BUILT** |
| E11 | Portable cache layer over any backend | Outline section 12-A1: a drift-correcting layer that sits on top of whichever cache is in use. | D | `DENOISE` | low | L | **RES** |
| E12 | MagCache (magnitude-aware cache) | `magcache` module (diffusers `MagCacheConfig`) with per-step magnitude ratios calibrated on the model (`calibrate_mag_ratios`; Wan's are bundled and picked up automatically for the calibrated setup). **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps; DINOv2 similarity to the unoptimized video (noise controls 0.88)): the best cache tried. Threshold 0.06: 1.49x, DINO similarity 0.885; 0.24: 2.03x, 0.855; both statistically the same as the noise control, 2-4 of 32 clips below the floor, prompt match unchanged.** Higher thresholds add nothing (max_skip_steps 3 and retention_ratio 0.2 bound it); max_skip_steps 5 gives 2.15x at 0.833; retention 0.1 collapses it (0.646). With CFG truncation 0.6 + KV cache + bf16 VAE: 2.54x at 0.856 (2.63x at 0.847 with threshold 0.5); the contact sheets keep the baseline scene on every prompt viewed, with a small measurable style shift (pixel deviation 0.12 against the control's 0.08). `recommended_config(model, aggressive=True)` returns this stack. | D | `DENOISE` | low | M | **BUILT** |

## F. Attention and tokens

Make the transformer's attention cheaper. Diffusion/AR transformers only; Dreamer has no attention.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| F1 | FlashAttention-2/3, SDPA backend selection | `attention_backend` module (diffusers backends). **Measured (Wan2.1-1.3B, 4 prompts x 1 seed, 33 frames at 192x320, 30 steps, baseline 12.4 s; fidelity = agreement with the unoptimized video.): native and memory-efficient SDPA identical (same kernel in practice); cuDNN SDPA 1.10x with a different output (PSNR 15.4: a numerically different kernel reaches the metric's noise floor of ~14-20 dB; not a bug); flex 0.34x (slower); flash SDPA unavailable here ('No available kernel').** No trustworthy win on this machine. | D, AR | `DENOISE` | none | S | **BUILT** |
| F2 | xFormers memory-efficient attention | Alternative fused attention. | D, AR | `DENOISE` | none | S | **DIFF** |
| F3 | FlexAttention custom masks | `attention_backend` with `flex`. **Measured (Wan2.1-1.3B, 4 prompts x 1 seed, 33 frames at 192x320, 30 steps, baseline 12.4 s; fidelity = agreement with the unoptimized video.): 0.34x (36.4 s vs 12.4 s) with PSNR 16.5.** Slower and different here; custom masks not explored. | D, AR | `DENOISE` | none | M | **BUILT** |
| F4 | Training-free sparse attention (Sparse VideoGen, Sparse-vDiT, DraftAttention, LoSA, HASTE, dynamic mixture-of-distributions) | Attention is >80% of latency per DraftAttention; these skip most of it without retraining. Several 2026 variants. | D | `DENOISE` | med | M | **DIFF** |
| F5 | Sliding-tile / windowed attention | Restrict attention to a local spatio-temporal window. | D | `DENOISE` | med | M | **DIFF** |
| F6 | Token merging / pruning (ToMe-style) | Merge similar tokens before attention. | D | `DENOISE` | med | M | **DIFF** |
| F7 | QuantSparse (quantization + sparsification combined) | Joint 4-bit-weight and sparse attention; reported ~15% attention density with little loss. | D | `DENOISE` | med | L | **DIFF** |
| F8 | FAST-AR (TempCache KV compression + ANN cross/self-attention) | For autoregressive video diffusion world models: compress the KV cache by temporal correspondence and select keys by approximate nearest neighbor; reported 5-10x end to end. | D, AR | `KV` | low | L | **DIFF** |
| F9 | Linear / low-rank attention conversion | Replace softmax attention with a cheaper form. Needs finetuning. | D, AR | `TRAIN` | high | L | **TRAIN** |

## G. Model compression

Make the model itself smaller.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| G1 | Speculative decoding for diffusion (SDVG) | Small draft model proposes, large model verifies (outline section 7.7, arXiv 2604.17397). | D | `DENOISE` | med | L | **DIFF** |
| G2 | Layer skipping / early exit | Skip layers on easy steps. Dreamer's layers are tiny, so little to gain there. | all | `TORCH` | med | M | **NOW** |
| G3 | Low-rank factorization of weights (SVD) | `low_rank` module. **Measured: harms physics badly on held-out walking** (skill 0.299 -> 0.112 at rank 0.25, 0.185 at 0.5; PSNR about -1.8 dB) despite a possible ~13% speedup (inside the noise; I had predicted it would be slower). Do not use on this model without finetuning. | all | `TORCH` | med | M | **BUILT** |
| G4 | Structured pruning (+ finetune) | Remove channels/heads; recover with finetuning. | all | `TRAIN` | med | L | **TRAIN** |
| G5 | Distill into a smaller student world model | Train a smaller Dreamer/DiT to imitate the large one. | all | `TRAIN` | med | L | **TRAIN** |
| G6 | Retrain with a smaller RSSM (dyn_deter 512 -> 256) | Cheaper per step by construction, at some quality cost. A config change plus a training run. | R | `TRAIN` | med | M | **TRAIN** |

## H. VAE / decoder side

Diffusion world models spend a lot of time encoding and decoding video.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| H1 | VAE tiling / slicing | `vae` module (`tiling=True`). **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps; DINOv2 similarity to the unoptimized video (noise controls 0.88)): output identical, 0.98x (slightly slower), peak GPU memory 4.15 GB against 4.40 GB.** A memory option only. | D | `DENOISE` | low | S | **BUILT** |
| H2 | Tiny VAE decoders (TAE-style) | Swap in a distilled lightweight decoder for previews or fast paths. | D | `DENOISE` | med | M | **DIFF** |
| H3 | Chunked / streamed decoding | Decode as frames are produced instead of at the end. | D, R | `STEPHOOK` | none | M | **DIFF** |
| H4 | Lower-precision VAE | `vae` module (`dtype="bfloat16"`). **Measured (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps; DINOv2 similarity to the unoptimized video (noise controls 0.88)): 1.04x, DINO similarity 0.999 (PSNR about 51 dB), peak GPU memory 3.58 GB against 4.40 GB.** A near-free win on speed and memory; part of the default diffusion stack. diffusers warns that Wan's VAE is recommended in fp32; the measured difference is negligible here. | D | `DENOISE` | low | S | **BUILT** |

## I. Resolution and temporal tricks

Generate less, then reconstruct the rest.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| I1 | Lower fps + frame interpolation | Same operation as C3 here (`sparse_decode`); same measured verdict. | all | `STEPHOOK` | med | M | **BUILT** |
| I2 | Keyframes + interpolation | Generate sparse keyframes, fill between. | D | `DENOISE` | med | M | **DIFF** |
| I3 | Low-res generate + super-resolution | Run the expensive model at low resolution and upscale. | D | `DENOISE` | med | M | **DIFF** |
| I4 | Hierarchical / progressive horizon rollouts | Coarse long-horizon plan first, refine near-term detail. | all | `none` | high | L | **RES** |
| I5 | Region-of-interest generation | Only simulate the parts of the scene a consumer needs. | all | `none` | high | L | **RES** |

## J. Memory

Fit bigger models or longer horizons; reduce held memory.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| J1 | VRAM check + automatic dtype / quantization selection | `worldoptbench.memory`: `recommend(param_count, vram_gb, compute_capability)` / `check_vram(info)` pick the highest-fidelity dtype that fits and warn when quantization or multi-GPU is needed. A rough weights-times-overhead estimate, unit-tested; not a module. | all | `TORCH` | none | M | **EXP** |
| J2 | CPU / sequential offload | Keep inactive weights in host memory. Only matters for big models. | D | `TORCH` | none | M | **DIFF** |
| J3 | Allocator tuning (expandable_segments) | Tested: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is **not supported on Windows** (PyTorch warns "expandable_segments not supported on this platform"). Would work on Linux. | all | `none` | none | S | **TOOL** |
| J4 | Chunked FFN / attention | Process in chunks to cut peak activation memory. | D, AR | `DENOISE` | none | M | **DIFF** |
| J5 | KV eviction + retrieval (WorldKV) | Keep evicted KV chunks in GPU/CPU memory and retrieve scene-relevant ones by camera/action correspondence (arXiv 2605.22718). | AR | `KV` | low | L | **DIFF** |
| J6 | Sliding window + attention sinks | Bound the KV cache for unbounded horizons. | AR | `KV` | med | M | **DIFF** |

## K. Parallelism and multi-GPU

All need more than one GPU (this machine has one).

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| K1 | Tensor parallelism (Accelerate) | Split weights across GPUs (outline section 7.10). | all | `TORCH` | none | M | **MULTI** |
| K2 | CFG parallelism | Run the two guidance branches on two GPUs. | D | `DENOISE` | none | M | **MULTI** |
| K3 | Sequence parallelism (Ulysses / Ring, xDiT USP) | Split the token sequence across GPUs. | D | `DENOISE` | none | L | **MULTI** |
| K4 | PipeFusion / patch pipelines | Pipeline image patches across GPUs. | D | `DENOISE` | low | L | **MULTI** |
| K5 | DistriFusion (displaced patch parallelism) | Reuse stale activations to hide communication. | D | `DENOISE` | low | L | **MULTI** |
| K6 | Pipeline parallelism | Different GPUs hold different layers (outline future work). | all | `TORCH` | none | L | **MULTI** |
| K7 | Spatial parallelism (Video-Infinity style) | Split frames spatially across GPUs. | D | `DENOISE` | low | L | **MULTI** |
| K8 | Data parallelism over rollouts | Throughput scaling: different rollouts on different GPUs. | all | `BATCH` | none | S | **MULTI** |
| K9 | Disaggregated stages (context encode / imagine / decode) | Separate GPUs per phase, as inference servers do for LLM prefill/decode. | all | `SERVER` | none | L | **MULTI** |

## L. Serving (worldserve)

Throughput and latency at the system level. Outline section 7.11.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| L1 | Continuous / dynamic batching | Admit requests into a running batch (vLLM-style). | all | `SERVER` | none | L | **NOW** |
| L2 | Horizon-bucketed scheduling | `scheduling.bucket_requests`: groups requests by (context frames, steps, frame size) into batches of at most `max_batch`, in first-seen order, returning indices so results can be scattered back. Pure Python, unit-tested. | all | `SERVER` | none | M | **EXP** |
| L3 | Streaming frame output | Return frames as produced; pairs with chunked graphs (A6). | all | `SERVER` | none | M | **NOW** |
| L4 | Result / context cache | `scheduling.ResultCache` (LRU, hit/miss counts, `namespace` so a different optimization stack can't return a stale result) and `request_key` (hash of frames, actions, horizon, seed). Pure Python, unit-tested; a request-level cache only — the context-latent cache (C5) is separate and not built. | all | `SERVER` | none | S | **EXP** |
| L5 | Async CPU/GPU pipeline | Overlap preprocessing, GPU work, and postprocessing across requests. | all | `SERVER` | none | M | **NOW** |
| L6 | Hardware auto-configuration | `defaults.recommended_stack(model)`: the best *measured* stack whose modules all apply here (tensorrt+graphs, else gumbel+graphs, else graphs, else no-validation, else nothing); never recommends an experimental module. Plus `memory.recommend` for dtype. Rule-based from one model's measurements; `autotune` is the principled version. | all | `SERVER` | none | M | **EXP** |
| L7 | Model multiplexing / lazy load-unload | Host several world models on one GPU. | all | `SERVER` | none | M | **NOW** |
| L9 | Library discovery and automatic stack selection (`library_report`, `autotune`) | Modules declare requirements and maturity; `library_report(model)` lists what applies; the stack skips non-applicable modules with a reason; `autotune` greedily builds the best stack by measured PAES. **First end-to-end run** (held-out walking episodes, 14 candidates, ~70 evaluations): chose `cuda_graphs` + `tensorrt` (fp16), PAES 3.95 vs 0.27 baseline, 3.5 ms mean — matching the sweeps — rejected `low_rank`/`latent_noise`/stacked `sparse_decode` on physics, and stopped when the best addition gained +1.4%. Known limitation: its physics guard needs a noise-aware tolerance (`--max-physics-drop 0.06`) for modules that change the random stream; at the default it would reject the best stack. | all | `none` | none | M | **BUILT** |
| L8 | Fast start (cached compiled artifacts and quantized weights) | Skip repeated quantization/compile at startup. | all | `SERVER` | none | S | **NOW** |

## M. Quality-preserving / long-horizon (raise PAES, not just speed)

These improve the physics axis, usually at a small speed cost. PAES rewards them.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| M1 | Drift correction on top of caching (outline's novel piece) | `AdaCacheModule(guard=...)`: a portable drift guard under a cache (outline 3.5 rescoped to option 12-A1). The first block runs as a per-step probe; scheduled reuse is vetoed when its drift since the last full computation exceeds the threshold. **Measured on AdaCache (Wan2.1-1.3B, 16 prompts x 2 seeds = 32 clips, 33 frames at 192x320, 30 steps, baseline 12.5 s; PSNR = agreement with the unoptimized video, mean +- sem over clips.): at matched speed it lowers the reference-free deviation (1.54x: 0.197 vs 0.318 unguarded; 2.22x: 0.522 vs 0.647) but stays behind WorldCache (1.65x: 0.126).** Built for AdaCache only; token-level selective recompute (the original 3.5 design) NOT built; the outline's 60-80%-of-the-physics-gap target is untestable without a physics score for Wan. | D | `DENOISE` | n/a | L | **BUILT** |
| M2 | Quantization-induced drift correction | Correct the temporal/physics degradation quantization causes. Outline section 12-A2, called the front-runner. Testable on Dreamer: build on B-series. | all | `STEPHOOK` | n/a | L | **NOW** |
| M3 | Periodic re-anchoring to real observations | Replace the latent with the posterior from a real or simulated frame every K steps (hybrid sim-in-the-loop). | R | `STEPHOOK` | n/a | S | **NOW** |
| M4 | Uncertainty-gated high-precision recompute | Run in low precision, rerun a step in fp32 only when uncertainty spikes. | R | `STEPHOOK` | n/a | M | **NOW** |
| M5 | Mixed-fidelity schedule | Cheap numerics early in the rollout, careful numerics where error compounds (or the reverse). | all | `STEPHOOK` | n/a | M | **NOW** |
| M6 | Best-of-N with a physics scorer | Sample several rollouts and keep the most plausible. Trades speed for physics. | all | `BATCH` | n/a | M | **NOW** |
| M7 | Mode (deterministic) latent sampling | `latent_noise(scale=0)`. **Measured: no reliable benefit** — clearly worse on the 515k model's long random-action horizons (skill 0.248 vs 0.298, same noise streams), within noise on held-out walking (0.237 vs 0.250). The mean-seeking expectation did not hold. | R | `none` | n/a | S | **BUILT** |
| M8 | Latent noise / unimix scaling | `latent_noise(scale)` with 0 < scale < 1. **Measured: no reliable benefit** (scale 0.5: 0.264 vs 0.298 on random-action horizons; 0.266 vs 0.250 on held-out walking; both inside the noise). | R | `none` | n/a | S | **BUILT** |
| M9 | Context window / memory compression for long horizons | Keep a bounded, compressed summary of history. | AR | `KV` | n/a | L | **DIFF** |
| M10 | Train the world model longer | Not an inference optimization, but the largest lever on skill. The current model has ~1.4% of Dreamer's default update budget. | R | `TRAIN` | n/a | L | **TRAIN** |

## N. Fine-tuning and offline (needs GPU hours)

Listed for completeness; most overlap items above.

| ID | Optimization | What it does / evidence | Applies | Hook | Physics risk | Effort | Status |
|---|---|---|---|---|---|---|---|
| N1 | Domain SFT / LoRA fine-tuning | Adapt a world model to a target domain (outline section 4). | all | `TRAIN` | n/a | L | **TRAIN** |
| N2 | Distributed fine-tuning (DeepSpeed / FSDP) | Multi-GPU fine-tuning for robotics customers (outline section 4). | all | `TRAIN` | n/a | L | **MULTI** |
| N3 | Pruning-aware / quantization-aware finetune loops | Recover quality after G4/B16. | all | `TRAIN` | low | L | **TRAIN** |

## What to build next

**Built and measured (42):** A1 CUDA graphs, A4 TensorRT engine for the RSSM step, A6 Chunked graphs (capture k steps, replay ceil(N/k) times), A9 channels_last memory format for conv encoder/decoder, A10 cuDNN benchmark / autotune, A11 Graph memory-pool sharing across shapes, A13 Disable torch.distributions argument validation, A14 TensorRT reduced-precision engines (fp16), A15 TensorRT for the decoder, B1 TF32 matmuls, B2 bf16 autocast, B3 fp16 autocast, B4 INT8 weight-only (TorchAO), B5 INT8 dynamic (activations + weights), B6 FP8 weight-only and dynamic (TorchAO), C1 Lean scan, C3 Sparse decoding, C4 Latent-only rollouts, C13 Gumbel-max sampling step (pure PyTorch), D1 Better ODE solvers (DPM-Solver++, UniPC), D2 Timestep schedule / shift tuning, D3 CFG truncation, D4 Cache or skip the unconditional CFG branch, E1 WorldCache, E2 AdaCache, E3 TeaCache, E4 FasterCache, E5 Pyramid Attention Broadcast (PAB), E6 TaylorSeer / feature forecasting, E7 Block-wise cache / layer skipping (DeepCache-style, Delta-DiT), E10 Text-embedding and cross-attention K/V cache, E12 MagCache (magnitude-aware cache), F1 FlashAttention-2/3, SDPA backend selection, F3 FlexAttention custom masks, G3 Low-rank factorization of weights (SVD), H1 VAE tiling / slicing, H4 Lower-precision VAE, I1 Lower fps + frame interpolation, L9 Library discovery and automatic stack selection (`library_report`, `autotune`), M1 Drift correction on top of caching (outline's novel piece), M7 Mode (deterministic) latent sampling, M8 Latent noise / unimix scaling.

**Built, unit-tested, not yet measured (6):** A12 Pre-capture graphs at startup (warm pool), C2 Batched rollouts, J1 VRAM check + automatic dtype / quantization selection, L2 Horizon-bucketed scheduling, L4 Result / context cache, L6 Hardware auto-configuration. Next step for all of these is a sweep on hardware to either confirm or retire them.

Of the **buildable now** items, the ones most likely to matter, in order:

1. **TensorRT for the decoder and encoder (A15), then reduced-precision engines (A14).** The step loop is down to ~11 ms of ~13 ms at a 20 s horizon, with decode ~5 ms the next biggest piece.
2. **The STEPHOOK extension point.** It unlocks C3-C6, M2-M5, M7-M8 and I1: roughly a dozen items. It is the single most valuable piece of infrastructure we don't have, and it needs chunked graphs (A6) to coexist with whole-rollout capture.
3. **M2 quantization-induced drift correction.** Your outline's section 12 names it the front-runner novel contribution and it is testable on Dreamer. Blocked on a better-trained model so the drift signal isn't noise (and quantization itself showed no speed benefit here, so the case for it is about quality, not speed).
4. **B10 sensitivity-guided mixed precision**, which answers which layers can be quantized without hurting physics.
5. **C2 batched rollouts**, also the foundation for L1 (continuous batching in worldserve). Note TensorRT/graphs are built for batch size 1 today.

What's *not* worth doing on this machine: every **DIFF** item (no diffusion model is running), every **MULTI** item (one GPU), and torch.compile / the fused kernel (no compiler toolchain). They become available if you get Cosmos access, use an open DiT, add a second GPU, or move to Linux.

## Sources

From this project's outline: WorldCache (arXiv 2603.06331 / 2603.22286), AdaCache (2411.02397), AccVideo (2503.19462), SDVG (2604.17397), OpenWorldLib (2604.04707).

Found by search on 2026-10-03 (summaries only):

- [WorldCache: Content-Aware Caching for Accelerated Video World Models](https://arxiv.org/html/2603.22286) — 2.3x on Cosmos-Predict2.5-2B
- [XPENG X-Cache world-model accelerator](https://www.xpeng.com/pressroom/news/019e0199813b9dd703de8a02822900a1) — reported 2.7x, training-free; also few-step distillation
- [Fast Autoregressive Video Diffusion and World Models with Temporal Cache Compression and Sparse Attention](https://arxiv.org/html/2602.01801v1) — TempCache, AnnCA, AnnSA
- [WorldKV: Efficient World Memory with World Retrieval and Compression](https://www.opentrain.ai/papers/worldkv-efficient-world-memory-with-world-retrieval-and-compression--arxiv-2605.22718/)
- [QuantSparse](https://arxiv.org/pdf/2509.23681), [LoSA](https://arxiv.org/pdf/2608.12032), [HASTE](https://www.alphaxiv.org/abs/2605.14513.md), [Sparse-vDiT](https://arxiv.org/html/2506.03065v1), [DraftAttention](https://arxiv.org/html/2505.14708v1), [dynamic sparse attention via mixture of distributions](https://arxiv.org/pdf/2601.11641)

Established methods listed without a link (TeaCache, FasterCache, PAB, DeepCache, TaylorSeer, ToMe, FlashAttention, SageAttention, SVDQuant, SmoothQuant, DistriFusion, PipeFusion, xDiT, DPM-Solver++, UniPC, DMD, vLLM batching) are from general knowledge of the literature.
