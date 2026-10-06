# Dreamer setup (as built on this machine, 2026-10-03)

Machine: RTX 5070 Ti Laptop (12 GB, Blackwell / compute capability 12.0), 32 GB RAM, Windows 11.

## Layout

| What | Where |
|---|---|
| Python 3.11.9 (winget, per-user) | `%LOCALAPPDATA%\Programs\Python\Python311` |
| venv for this project | `ai\.venv` (git-ignored) |
| dreamerv3-torch clone | `Documents\dreamerv3-torch` (outside this repo) |
| Trained checkpoint(s) | `Documents\dreamerv3-torch\logdir\<name>\latest.pt` |

Activate: `.\.venv\Scripts\Activate.ps1` (or call `.\.venv\Scripts\python.exe` directly).

## Packages, and why they differ from the repo's `requirements.txt`

The repo pins `torch==2.4.1`, which predates Blackwell support — it cannot use this GPU. So:

- `torch 2.11.0+cu128`, `torchvision` — from `https://download.pytorch.org/whl/cu128` (CUDA 12.8 is the minimum for sm_120).
- `numpy 1.26.4` — torch pulls numpy 2.x; `gym 0.22` needs 1.x.
- `mujoco 2.3.5`, `dm_control 1.0.9`, `gym 0.22.0`, `ruamel.yaml 0.17.4` — the repo's own pins, kept (they work together).
- `tensorboard`, `cloudpickle`, `torchmetrics` (PSNR/SSIM), and `pip install -e .` for this project.
- `tensorrt-cu12 11.3.0.99`, `onnx 1.23.1`, `ml_dtypes 0.5.4`, installed with `numpy==1.26.4` pinned (their default resolution wants numpy 2.x, which breaks `gym 0.22`). **Do not install `torch-tensorrt`**: 2.14 pins torch 2.14 and would replace the working torch 2.11.
- No `nvcc` and no host C++ compiler (MSVC) on this machine, so custom CUDA extensions and `torch.compile`/Triton can't be built. CuPy's NVRTC route could compile CUDA kernels without a host compiler if ever needed.
- Deliberately **not** installed: `matplotlib==3.5.0` (no Python 3.11 wheel), `moviepy` (only for logging videos), `opencv`, `memory_maze`, `crafter` (other environments).

## Local patch to dreamerv3-torch

`dreamer.py` line 7 hardcoded `MUJOCO_GL="osmesa"`, which doesn't exist on Windows. Changed to
`os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "win32" else "osmesa")`.
(`worldoptbench.models.dreamer.DreamerRepo` handles the same issue when it imports the repo.)
If you re-clone, reapply it.

## Training

```powershell
cd Documents\dreamerv3-torch
..\ai\.venv\Scripts\python.exe dreamer.py --configs dmc_vision --task dmc_walker_walk `
  --logdir .\logdir\walker --steps 100000 --prefill 2500 --eval_every 10000 `
  --eval_episode_num 1 --log_every 5000 --envs 1 --video_pred_log False --train_ratio 128
```

- Measured ~16-19 env steps/s at `--train_ratio 128` (~2.4 model updates/s) with `compile` off, fp32.
- `latest.pt` is rewritten every `--eval_every` steps, so a partly-trained checkpoint is usable (avoid reading it mid-write).
- A 5,000-step smoke run (`logdir\smoke`) takes ~5 min and produces a model that predicts blurry blobs — enough to test plumbing, not to benchmark.
- Resuming: rerun the same command; it loads `latest.pt` and continues.

## Verified on this machine

- CUDA works on the 5070 Ti with torch 2.11 (matmul smoke test).
- `dm_control` walker renders 64×64 with `MUJOCO_GL=glfw`; control timestep is 0.025 s (so 20 agent steps/s at action_repeat 2 — matches `DreamerRepo.steps_per_second`).
- `DreamerRepo` config/env loading, `DreamerWorldModel.generate()`, and `DreamerSimScenarios` all ran correctly against the smoke checkpoint.

- `torchao 0.18.0` (installed `--no-deps` so it can't swap torch) works with torch 2.11: all config classes exist and
  int8/float8 weight-only + int8/float8 dynamic quantization ran on the Dreamer model. `int4_weight_only` needs `mslk >= 1.0.0` (not installed).
- TensorRT 11.3 builds and runs engines on the 5070 Ti (the RSSM-step engine builds in ~5 s; cached under `~/.cache/worldoptbench/trt`).
- `py-spy` is installed in the venv for diagnosing slow runs (`py-spy dump --pid <pid>`).

## Gotchas learned

- **Seed everything you compare.** Dreamer samples its latent state, so two *unseeded* generations from identical weights differed by ~12/255 pixels
  and their fidelity scores ranged 0.06-0.27 — bigger than any optimization effect. `DreamerWorldModel.generate(seed=...)` now seeds (and restores) the RNG;
  `DreamerSimScenarios` passes a per-scenario seed. Same seed gives essentially identical output (rare +-1/255 pixel jitter from non-deterministic GPU kernels appears even between identical runs, so don't expect literal bit-equality).
- **Modern Standby silently suspends training.** This laptop enters Modern Standby whenever the display goes off, and Windows then suspends background desktop processes: a 27-minute "episode" and later a ~3h40m stall were both this (the event log shows repeated `Entering Modern Standby`). `SetThreadExecutionState(ES_SYSTEM_REQUIRED)` does **not** prevent it, and a power request only protects the process that made it. `scripts/train_dreamer.py` therefore runs dreamer.py *in-process* while holding execution-required + system-required + display-required power requests (releasing on exit; no system settings changed). Use it for any long run, keep the laptop plugged in, and dim the screen rather than turning it off. It keeps the display on by default; `--allow-display-off` relies on the execution request alone, which is untested over hours.
- Python on Windows: pass `encoding="utf-8"` to `read_text`/`write_text` (default cp1252 corrupted a file once).

## Results (2026-10-03, trained walker `logdir\walker`, 115k env steps, GPU otherwise idle)

12 rollouts = 3 scenarios x horizons 2/5/10/20 s. Reproduce with `python scripts/sweep_dreamer.py --checkpoint ..\dreamerv3-torch\logdir\walker` (17 configurations x 3 repeats, ~11 min). Raw files in `results\sweep\` (git-ignored).

### How to read the numbers

- **Compare absolute latencies, not speedup ratios.** The eager baseline is CPU-launch-bound and its run-to-run noise is huge: it measured 210-284 ms across identical runs, and the sweep's own three repeats spread +-34%. Anything divided by it inherits that. The CUDA-graph configurations are far steadier (0-15%). The `noise` figure is half the range of the per-run means as a % of the median; treat gaps smaller than it as unproven.
- Earlier numbers (5.5x, then 11.5x for CUDA graphs) were taken with a methodology that flattered speedups; the table below supersedes them. Three things were wrong, all fixed in the runner: (1) scenario building (a fresh OpenGL simulator env) ran right before each timed call and made the eager baseline ~2x slower; (2) PSNR/SSIM/physics ran *between* timed calls and slowed the next TensorRT-in-graph call by ~5 ms (30.5 vs 25.8 ms at 20 s) while barely touching PyTorch ones; (3) warm-up only covered the shortest horizon. The runner now builds all scenarios first, warms up every horizon, times all rollouts back to back, and only then computes metrics.

### Sweep (median of 3 runs per configuration)

| configuration | latency | noise | speedup vs eager | physics skill | GPU memory held |
|---|---|---|---|---|---|
| baseline (eager) | 209.9 ms | 34% | 1.00 | 0.253 | 544 MB |
| cuda_graphs | 21.7 ms | 15% | 9.4x | 0.253 | 672 MB |
| lean_scan + cuda_graphs | 21.1 ms | 1% | 9.6x | 0.253 | 668 MB |
| tf32 + cuda_graphs | 21.6 ms | 11% | 9.4x | 0.253 | 672 MB |
| lean + tf32 + cuda_graphs | 21.0 ms | 1% | 9.6x | 0.253 | 668 MB |
| bf16 + cuda_graphs | 31.5 ms | 7% | 6.5x | 0.249 | 570 MB |
| fp16 + cuda_graphs | 27.0 ms | 12% | 7.7x | 0.256 | 570 MB |
| int8 weight-only + cuda_graphs | 23.1 ms | 15% | 8.8x | 0.250 | 730 MB |
| **gumbel (pure PyTorch) + cuda_graphs** | **17.6 ms** | 0% | 11.4x | 0.275* | 668 MB |
| **tensorrt + cuda_graphs** | **13.2 ms** | 1% | **14.8x** | 0.275* | 668 MB + TensorRT workspace |
| eager only: lean_scan / tf32 / bf16 / int8 / no_dist_validation | 232-278 ms | 2-41% | 0.76-1.00 (within noise) | unchanged | 438-544 MB |
| eager only: gumbel | 146.9 ms | 26% | 1.6x | 0.275* | 544 MB |
| eager only: tensorrt | 35.2 ms | 13% | 5.3x | 0.275* | 544 MB + workspace |

\* Gumbel and TensorRT draw different random numbers than the baseline (see finding 3), so individual rollouts differ and the mean over only 12 differs by sampling noise (+0.022 here; per-rollout noise alone is +-0.05). They are statistically equivalent: over 180 rollouts each, pooled skill was 0.274 (Gumbel) vs 0.276 (repo sampler), z = -0.36.

### Findings

1. **CUDA graphs are the big lever** (~9x here): the repo's imagination loop is launch-bound (a Python loop of ~70 tiny kernels per frame, GPU mostly idle). Fusing the whole rollout (context + imagination + decode) into one graph required rewriting the repo's `obs_step`, which host-syncs on a tensor-valued `if`; the rewrite reproduces the repo's output (PSNR difference 0, skill difference 1e-7).
2. **TensorRT is the best configuration: 13.2 ms, 1.33x faster than the pure-PyTorch Gumbel step in graphs and 1.64x faster than the repo's loop in graphs** (all three at ~1% noise; in-process steady state at 20 s: 23.8 vs 31.3 ms). In isolation the loop is 28 us/step vs 59 us/step. It reproduces the pure-PyTorch Gumbel reference (same noise, same math) to 2.5e-6 in skill and 1.1e-5 in PSNR through the full pipeline, and sampled states agree 100% over 400 chained steps.
3. **Why a different sampler:** TensorRT has no RNG and can't express `torch.multinomial`. Sampling is Gumbel-max with noise drawn up front (same distribution, different random stream). `gumbel_sampling` exposes the same math in plain PyTorch, so TensorRT can be checked against an exact reference; it is also a speedup on its own (17.6 vs 21.7 ms with graphs).
4. **Quantization (int8/fp8) and mixed precision (bf16/fp16) don't help** on this model: with graphs they are slower (23-31 ms vs 21.7), physics shifts slightly (bf16 -0.004, fp16 +0.003), and int8 costs +58 MB held. bf16 saves ~100 MB. This is the expected outcome for a tiny launch-bound network.
5. **lean_scan and tf32: no demonstrated effect.** With graphs they land at 21.0-21.7 ms against 21.7 (inside the noise). An earlier sweep suggested ~4% and the repeat sweep did not reproduce it, so that claim is retracted.
6. **no_dist_validation:** a controlled in-process A/B measured **1.43x on the eager path** (431 -> 302 ms at 20 s; 53 -> 38 ms at 2 s), because torch.distributions' argument checks each cost a GPU->CPU sync. The sweep can't confirm it (eager noise 31%). No effect with graphs, which already disable validation during capture.
7. **Memory:** graphs hold +128 MB (544 -> 672 MB; `vram_reserved_gb` shows it, `vram_peak_gb` doesn't). TensorRT also allocates a workspace outside PyTorch's allocator that no column here counts.
8. **Where the time goes now (TensorRT + graphs, 20 s horizon):** the 400-step loop ~11 ms, decode ~5 ms (13 us/frame), context ~3 ms, copies/Python ~0.4 ms. The "~20 ms host-overhead floor" suggested earlier was a pre-fusion artifact; host overhead is negligible.
9. **Not feasible here:** `torch.compile` (needs Triton + MSVC).

### The world model itself is mediocre

Skill ~0.25 (1.0 = perfect, 0 = freeze the scene), teacher-forced reconstruction only 23 dB, barely better than a freeze-the-scene predictor, from ~7k gradient updates (~1.4% of Dreamer's default budget). Not a pipeline bug: the score on the model's own training episodes was no better, the current action alignment beat both +-1 shifts, and zeroing the actions cut skill 0.18 -> 0.06. **Drift is not measurable yet:** per-seed drift rates have mixed signs (-0.006, +0.001, +0.005 /s) and mean skill is flat across horizons; it needs a better-trained model and more seeds before PAES's drift term means anything.

## Not yet done

- A better-trained world model (resume `logdir\walker` toward ~500k steps; ~1.3k env steps/min => ~5 h) so drift/PAES comparisons are meaningful.
- Batched rollouts, a per-step hook (early stop, re-anchoring, drift correction), chunked graphs; see `OPTIMIZATION_CATALOG.md`.
- TensorRT for the decoder (5 ms of the remaining 13 ms) and reduced-precision TensorRT engines.

## Results on the 515k-step model (2026-10-03, `logdir\walker_long`)

Trained from the 115k checkpoint to 515k env steps (~32k gradient updates, still a small fraction of Dreamer's default budget). Eval return rose from ~440 to ~740-795. Same standard set and 3-repeat methodology as above; raw files in `results\sweep_long\`.

### Speed: the ranking reproduced

The speed results do not depend on the checkpoint, and they didn't move:

| configuration | latency (115k model) | latency (515k model) | noise |
|---|---|---|---|
| cuda_graphs (repo loop) | 21.7 ms | 21.0 ms | 1% |
| gumbel + cuda_graphs | 17.6 ms | 17.7 ms | 1% |
| **tensorrt + cuda_graphs** | **13.2 ms** | **12.9 ms** | 0% |
| bf16 / fp16 / int8 + cuda_graphs | 31.5 / 27.0 / 23.1 ms | 29.0 / 30.7 / 27.3 ms | 5-11% |

Same story as before: TensorRT + graphs is best (~1.6x the repo loop in graphs, ~16x the noisy eager baseline); quantization and mixed precision are slower than plain graphs and cost physics-score noise, not speed. lean_scan and tf32 again show no effect. Eager-baseline noise was 32% again, so ratios against it remain approximate; absolute latencies are the number to trust.

### Physics: three findings

1. **Gumbel / TensorRT sampling is still statistically equivalent to the repo's sampler.** Pooled over 180 rollouts each on this checkpoint: 0.318 (Gumbel) vs 0.310 (repo), z = 1.28, so no detectable difference (and on the 115k model, 0.274 vs 0.276). The sweep's +0.038 on 12 rollouts was sampling noise. A small effect under ~0.02 can't be excluded.
2. **More training barely moved the sweep's skill number** (0.260 vs 0.253) despite a far better policy. The reason is below.
3. **Drift is still not measurable:** mean drift rate -0.0026 +- 0.0036 per second over 3 scenarios, "not distinguishable from 0", for every configuration.

### The benchmark's scenarios were measuring the wrong thing

Scored against *held-out episodes of the trained policy walking* (the `eval_eps` the model never trained on, return > 600), versus the benchmark's random-action scenarios, with skill = 1 - (model error / freeze-the-scene error):

| scenario type | horizon | 115k model | 515k model |
|---|---|---|---|
| held-out competent walking | 2 s | 0.042 +- 0.011 | 0.086 +- 0.017 |
| held-out competent walking | 5 s | 0.018 +- 0.006 | 0.020 +- 0.008 |
| random actions (the benchmark's) | 2 s | 0.329 +- 0.031 | 0.354 +- 0.035 |
| random actions (the benchmark's) | 5 s | 0.323 +- 0.028 | 0.380 +- 0.026 |

- **Random actions flatter the model.** A flailing walker falls over and lies still, which is easy to predict. On competent walking, open-loop prediction is barely better than freezing the scene after 2 s, for both models.
- **Extra training helped at 2 s (0.042 -> 0.086) and not at 5 s.** A likely (unverified) explanation: DreamerV3 is trained to imagine ~15 steps (under a second) ahead from a posterior state, not to free-run for seconds, so long-horizon drift is saturated near zero and more training at this scale won't produce an informative 20 s drift curve. The informative range is sub-second to a few seconds.
- **Consequence:** the 20 s horizons say nothing about physics here (they still matter for *speed*, which is how long rollouts scale). For physics and drift, use `DreamerReplayScenarios` with held-out policy episodes and short horizons:

```powershell
.\.venv\Scripts\python.exe scripts\sweep_dreamer.py --checkpoint $ck --standard-set worldoptbench\prompts\dreamer_policy_set.json `
    --episodes ..\dreamerv3-torch\logdir\walker_long\eval_eps --min-return 600 --out-dir results\sweep_long_policy
```

(12 scenarios x horizons 0.25-5 s; 16 eligible held-out episodes on this checkpoint.) Caveat: the eval episodes come from the same policy that generated the training data, so they test generalization to new trajectories of the same behaviour, not to novel behaviour.

### The experimental modules, measured (515k model, 3-scenario random-action set, 3 repeats)

Every row is compared against `tensorrt + cuda_graphs` (13.1 ms, 1% noise), except the repo-loop rows, which are compared against `cuda_graphs` alone (21.0 ms from the standard sweep). `noise` is the run-to-run half-range; gaps smaller than it are unproven. Output comparisons use identical noise, so they isolate the module's own effect.

| configuration | latency | noise | vs reference | physics vs reference | verdict |
|---|---|---|---|---|---|
| **tensorrt fp16 + graphs** | **9.9 ms** | 1% | **-24% (1.32x faster)** | mean skill diff 0.007, PSNR within 0.2 dB (sampling noise alone is ~0.05) | **new best; real**. Faster at every horizon (20 s: 19.2 vs 26.2 ms) |
| + tensorrt_decoder | 12.0 ms | 4% | -8% | **identical** (0 difference in skill and PSNR) | correct, modest gain; its "338 MB held" is not a saving (TensorRT workspace isn't counted by PyTorch) |
| + sparse_decode 2 / 4 / 8 | 11.9 / 11.4 / 11.1 ms | 0-1% | -9 / -13 / -15% | skill +0.006 / +0.009 / +0.004; PSNR +0.2 / +0.3 / +0.3 dB | faster, but see the caveat below |
| + tensorrt_decoder + sparse 4 | 11.2 ms | 1% | -15% | as sparse 4 | the decoder engine adds ~nothing once decoding is already thinned 4x |
| + latent_noise 0 (mode) / 0.5 | 12.9 / 12.9 ms | 0-1% | no speed effect | skill **0.248 / 0.264 vs 0.298**; per-rollout diff mean 0.05, max 0.12 | **worse**: deterministic mode rollouts compound into unrealistic states |
| cuda_graphs + share_pool | 27.5 ms | 13% | +31% slower (vs 21.0) | identical | saves **172 MB** held (672 -> 500 MB, -26%) but looks slower; worth re-measuring, the noise is high |
| cuda_graphs + cudnn_benchmark | 20.9 ms | 6% | no effect | identical | no effect (tiny convs) |
| cuda_graphs + channels_last | 23.6 ms | 13% | no gain | identical | no gain |
| cuda_graphs + low_rank 0.25 | 18.3 ms | 11% | -13% (inside the noise) | skill -0.006, PSNR -0.4 dB | **surprise**: I predicted slower (extra kernels); instead likely faster, because the weight-reading GEMVs are memory-bound and the factors halve the bytes read. Inconclusive at this noise level; lossy |
| cuda_graphs + low_rank 0.5 | 21.4 ms | 10% | no gain | skill +0.011, PSNR -0.2 dB | no gain |

**Caveat on sparse_decode (and any blur):** interpolated frames are cross-fades, and a pixel-error metric *rewards* blur (skill and PSNR rose). The speed gain is real; the physics "improvement" is a metric artifact, not a better prediction. Don't count it as a PAES win without a metric that penalizes blur (motion/state-based, or sharpness-aware).

**What these say about the project:**
- **fp16 on the TensorRT step is a real win** where fp16/bf16 autocast in PyTorch was a loss: the step is latency-bound, and the engine fuses it, so lower-precision arithmetic shortens the kernels instead of adding casts.
- The remaining time at 20 s is now mostly the step loop (~11 ms of fp32 ~13 ms); decode is the next piece and `sparse_decode` / `tensorrt_decoder` trade against it.
- **Mode sampling is not a free quality trick.** It was expected to lower pixel error; on this model it raises it.

### Physics on realistic data (held-out walking episodes, short horizons)

`dreamer_policy_set.json`: 12 scenarios from held-out `eval_eps` (return > 600) x horizons 0.25 / 0.5 / 1 / 2 / 3 / 5 s, 3 repeats, `--episodes ...\eval_eps --min-return 600`. Raw files in `results\sweep_long_policy\`. Eager baseline (repo sampler, no graphs): skill 0.299, PSNR 19.58.

**Drift is finally measurable: -0.092 +- 0.009 skill per second** (detectable) for the baseline, on the one data type where the model has signal. On random-action scenarios it was -0.003 +- 0.004 (not distinguishable from zero), which is why drift looked unmeasurable before.

| configuration | latency | skill | vs same-noise reference | verdict |
|---|---|---|---|---|
| baseline (eager) | 56.1 ms | 0.299 | | |
| tensorrt + graphs (reference) | 4.1 ms | 0.250 | | the sampler differs from the repo's: see "Reading the physics column" |
| **tensorrt fp16 + graphs** | **3.4 ms** | 0.251 | +0.001 | **-17% latency, same physics** |
| + tensorrt_decoder | 3.7 ms | 0.250 | **0.000 (identical)** | -10% latency, same output |
| + sparse_decode 2 / 4 / 8 | 4.1 / 3.9 / 3.8 ms | 0.236 / 0.203 / 0.171 | **-0.014 / -0.047 / -0.079** | quality loss grows with stride |
| + latent_noise 0 / 0.5 | 4.1 / 4.3 ms | 0.237 / 0.266 | -0.013 / +0.016 | inside the noise |
| cuda_graphs + low_rank 0.25 | 5.1 ms | **0.112** | -0.187 (vs baseline) | **wrecks physics** |
| cuda_graphs + low_rank 0.5 | 5.7 ms | **0.185** | -0.114 (vs baseline) | **wrecks physics** |
| cuda_graphs + share_pool / cudnn_benchmark / channels_last | 6.0 / 6.1 / 6.1 ms | 0.299 | 0.000 (identical) | no effect on output |

Where the **same noise** is used (decoder, decimation, weights, precision), the comparison is tight and the differences above are real.

### Reading the physics column: effective sample size is the number of scenarios

The sweep's `d_skill` for configurations that **change the random stream** (Gumbel / TensorRT vs the repo sampler, and `latent_noise`) is **underpowered**: the 6 horizons of one scenario are nested prefixes that share their random draws, so 72 rollouts are only ~12 independent samples (standard error ~0.02-0.03 on a difference). The apparent -0.049 for TensorRT vs the baseline here (and +0.038 on the random-action set) were both noise. The direct test, with independent noise draws per rollout, on these held-out episodes (6 scenarios x 40 seeds x 2 horizons = 480 rollouts per sampler):

| sampler | skill 1 s | skill 3 s | pooled |
|---|---|---|---|
| repo (`torch.multinomial`) | 0.376 +- 0.011 | 0.118 +- 0.007 | 0.247 |
| Gumbel-max, pure PyTorch | 0.370 +- 0.011 | 0.111 +- 0.007 | 0.240 (diff -0.007, z = -0.56) |
| Gumbel-max, TensorRT fp32 | 0.370 +- 0.011 | 0.111 +- 0.007 | 0.240 (identical to the PyTorch Gumbel step) |

So the Gumbel/TensorRT sampler is statistically equivalent to the repo's on three checkpoint/scenario combinations now (115k random-action: 0.274 vs 0.276; 515k random-action: 0.318 vs 0.310; 515k held-out walking: 0.240 vs 0.247), and TensorRT reproduces the pure-PyTorch step exactly. A bias under ~0.01-0.02 can't be excluded. **Rule of thumb: to detect a physics effect of a module that changes the noise, use many independent seeds per scenario, not more horizons.**

### Autotune, first end-to-end run

```powershell
scripts\autotune_dreamer.py --checkpoint $ck --standard-set worldoptbench\prompts\dreamer_policy_set.json `
    --episodes ..\dreamerv3-torch\logdir\walker_long\eval_eps --min-return 600 `
    --repeats 1 --max-modules 5 --max-physics-drop 0.06 --module-kwargs "@results\autotune_kwargs.json"   # {"tensorrt":{"precision":"fp16"}}
```

14 candidate modules, ~70 evaluations, `--repeats 1` (a rough first run). Result: **`['cuda_graphs', 'tensorrt']` at fp16, PAES 3.95 against a baseline of 0.27, 3.5 ms mean latency.** The search reproduced what the hand-run sweeps found:

- It took `cuda_graphs` first (PAES 3.5), then `tensorrt` on top (+12%, accepted); the best remaining addition (`lean_scan`) gained +1.4%, under the 10% margin, so it stopped.
- It **rejected on physics**: `low_rank` (0.112, -0.19), `latent_noise` (0.234), stacked `sparse_decode` (0.204) and stacked `quantization` (0.230).
- It correctly scored `tensorrt` followed by `gumbel_sampling` *worse* (the later module replaces the faster backend with the slower pure-PyTorch one).
- With one repeat, near-ties are noise: `tensorrt_decoder` scored below the accepted stack here although the sweeps measured it ~10% faster; it would have needed `--repeats 3` to separate.

**Limitation to know before using it elsewhere:** the TensorRT stacks scored physics 0.251 against the baseline's 0.299 purely from sampling noise (see "Reading the physics column"), so at the default `--max-physics-drop 0.02` the best stack would have been rejected. Set the tolerance to ~2 standard errors for your scenario count (~0.06 for 12 scenarios) or use many independent seeds. `defaults.recommended_config` encodes the same stack (with the fp16 kwarg) for when you don't want to run the search.

## Runbook: after the long training run finishes

The long run trains `logdir\walker_long` (a copy of the 115k reference; `logdir\walker` is untouched). When it's done and the GPU is idle:

```powershell
cd Documentsi
$ck = "..\dreamerv3-torch\logdir\walker_long"

# 1. Standard sweep on the better model (the measured configurations, ~12 min)
.\.venv\Scripts\python.exe scripts\sweep_dreamer.py --checkpoint $ck --out-dir results\sweep_long

# 2. The experimental (unmeasured) modules, each stacked on the best measured stack (tensorrt + cuda_graphs)
.\.venv\Scripts\python.exe scripts\sweep_dreamer.py --checkpoint $ck --experimental --out-dir results\sweep_long_exp

# 3. Make drift measurable: 10 scenarios x 7 horizons (the table prints mean drift +- standard error per configuration)
.\.venv\Scripts\python.exe scripts\sweep_dreamer.py --checkpoint $ck --only "tensorrt + cuda_graphs" "cuda_graphs" --standard-set worldoptbench\prompts\dreamer_dmc_set_10seeds.json --out-dir results\sweep_long_drift

# 4. First end-to-end run of autotune (slow; --repeats 1 for a rough first look)
.\.venv\Scripts\python.exe scriptsutotune_dreamer.py --checkpoint $ck --repeats 2
```

Then mark each experimental module `measured` or retire it (and update `OPTIMIZATION_CATALOG.md`). The new checkpoint is a different model from the 115k one (the old model's skill was ~0.25 and its drift was unmeasurable), so don't compare skill numbers across the two without saying so.

## Long-horizon scaling (2026-10-04, `scripts/bench_long_horizon.py`)

The outline asks for 60 s and 120 s horizons. DMC walker episodes end at 500 agent steps (25 s), so the simulator cannot provide ground truth that far and **no physics or drift score exists beyond 20-25 s on this task**; what can be measured is whether cost scales linearly and what memory the graphs hold. Synthetic inputs (zero context frames, random actions), median of 5 calls after warm-up, 515k-step model, RTX 5070 Ti laptop GPU, GPU otherwise idle:

| horizon | steps | eager (unoptimized) ms | per simulated second | TensorRT fp16 + CUDA graphs ms | per simulated second | graphs hold (MB, whole process) |
|---|---|---|---|---|---|---|
| 2 s | 40 | 53.7 | 26.9 | 3.3 | 1.63 | 149 |
| 5 s | 100 | 116.5 | 23.3 | 5.6 | 1.11 | 160 |
| 10 s | 200 | 221.7 | 22.2 | 9.4 | 0.94 | 171 |
| 20 s | 400 | 437.2 | 21.9 | 17.8 | 0.89 | 185 |
| 60 s | 1200 | 1271.4 | 21.2 | 54.8 | 0.91 | 209 |
| 120 s | 2400 | 2567.6 | 21.4 | 99.2 | 0.83 | 246 |

Both configurations scale linearly (the eager one at about 21 ms of compute per simulated second, the optimized one at about 0.85-0.9 ms after the fixed per-call cost is amortized), every step returns a frame, and the optimized stack is about 26x faster at 120 s. The graphs' persistent memory grows slowly with the horizon (about +1 MB per simulated second beyond the first few). A 120 s rollout in 99 ms is far past real time (the walker runs at 20 steps per second of simulated time); this says nothing about quality at that length, which cannot be scored here.

## Functional utility: does an optimized model still rank policies correctly? (2026-10-04, `scripts/policy_ranking_dreamer.py`)

The outline's section 12-D: a world model is often used to *evaluate policies* (imagine each candidate and rank by predicted return), and speed, visual and physics scores can all survive an optimization while that use quietly breaks. Policies are variants of the trained actor (`worldoptbench/models/dreamer_policy.py`; an actor from another checkpoint lives in a different latent space, so it cannot be evaluated inside this model). Two sets of nine: **coarse** (the actor, the actor with action noise 0.2 / 0.5 / 1.0, halved, held for two steps, inverted, zero, random: terrible to good play) and **fine** (the actor with noise 0 to 1 in nine steps, true returns close together). True return: real simulator episodes (6 per policy for coarse, 8 for fine), the baseline world model doing state estimation as the agent does. Imagined return: 24 held-out walking start states from recorded episodes, 60 imagined steps, the reward head's prediction summed. Agreement is Spearman rank correlation (`worldoptbench/utility.py`; kendall and pairwise are in `results/policy_ranking*/report.md`).

| model | coarse: vs simulator | coarse: vs unoptimized | fine: vs simulator | fine: vs unoptimized | actor's imagined return |
|---|---|---|---|---|---|
| unoptimized | +0.92 | | +0.95 | | 112 |
| unoptimized, different random draws (the noise floor) | +0.97 | +0.97 | +0.98 | +0.98 | 112 |
| Gumbel sampler | +0.92 | +1.00 | +0.95 | +1.00 | 112 |
| TensorRT fp16 | +0.92 | +1.00 | +0.95 | +1.00 | 113 |
| bf16 autocast | +0.92 | +1.00 | +0.98 | +0.98 | 112 |
| int8 weight-only | +0.92 | +1.00 | +0.95 | +1.00 | 113 |
| latent noise 0.5 / 0 | +0.97 / +0.97 | +0.97 / +0.97 | +0.97 / +0.98 | +0.97 / +0.98 | 106 / 99 |
| low-rank 0.5 | +0.93 | +0.98 | +0.98 | +0.98 | 100 |
| low-rank 0.25 | +0.93 | +0.95 | +0.97 | +0.97 | 40 |

What it shows, and does not:

- **The unoptimized world model is a good policy evaluator** (Spearman 0.92-0.95 against the real simulator), so the check is meaningful here.
- **No optimization measurably changed the ranking.** Every row agrees with the unoptimized model about as well as the unoptimized model agrees with itself under different random draws (0.97-0.98); TensorRT fp16, int8, the Gumbel sampler and bf16 reproduce its order exactly or with one swapped pair; the best policy is the same in all but one row (latent noise 0.5 on the fine set, where it swaps the two best of nine, true returns 824 and 801).
- **Ranking utility and physics fidelity can part ways, and absolute value is a separate matter.** Low-rank 0.25 lost most of its physics skill (0.11 against 0.30 in the Dreamer table) and its imagined returns collapse (the actor scores about 40 against 112), yet it still orders the policies at about 0.95-0.97. It remains usable to *rank* candidates and is useless for estimating how good one is. This is the case the outline's fourth axis was meant to catch: a physics number alone would have rejected it for a use it still serves.
- **Limits.** Nine policies cannot resolve small losses: one swapped pair moves Spearman by 0.02-0.03, so rows within a few hundredths of the reseed floor are indistinguishable from it. The policies are variants of one actor on one task, and several of them are effectively tied in true return (noise 0.2 / 0.3 / 0.4: 644 / 663 / 641 in the fine set, within the episode-to-episode spread), so no model could order those; the bottom of the coarse set (random, inverted, zero: 45 / 44 / 20 true, 15 / 11 / 15 imagined) is near-tied as well. A harder test needs more policies that differ a little, and more start states. Imagined returns come from walking start states over 3 s, not whole episodes.

## Latent-only rollouts (2026-10-04, `scripts/bench_latents.py`, catalog C4)

`DreamerWorldModel.generate_latents(...)` returns the imagined rollout as latent features (N, F) and skips the decoder, for consumers that never look at pixels (a planner or policy evaluator reads reward and value heads off the latents). Same inputs, seeds and optimization knobs as `generate()`. Median of 15 calls, 515k-step model:

| configuration | horizon | frames (ms) | latents (ms) | saved |
|---|---|---|---|---|
| CUDA graphs | 2 / 5 / 10 / 20 s | 5.95 / 12.31 / 22.92 / 44.65 | 4.90 / 10.10 / 19.12 / 36.87 | 18% / 18% / 17% / 17% |
| TensorRT fp16 + CUDA graphs | 2 / 5 / 10 / 20 s | 3.27 / 5.70 / 9.94 / 18.50 | 2.51 / 3.91 / 6.18 / 10.81 | 23% / 31% / 38% / 42% |

The decoder's share grows once the dynamics step is fast, so the saving is largest on the best stack. The eager rows (noisy, with negative "savings" at long horizons) are within that path's known run-to-run noise and are not reported.
