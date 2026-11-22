# V-JEPA 2: latent forecasting under optimization

Measured 2026-11-22 on an RTX 5070 Ti Laptop GPU (12 GB), WSL2 Ubuntu, torch 2.11 + CUDA 12.8. Code: `worldoptbench/models/vjepa2.py`, `worldoptbench/metrics/latent.py`, `worldoptbench/optimizations/torch_compile.py`, `scripts/sweep_vjepa2.py`. Raw results: `results/vjepa2/` (`summary.json` has every per-clip number).

## What this is, and what it is not

V-JEPA 2 predicts *representations*; it has no decoder, so the frame-based visual and physics metrics (PSNR, SSIM, the Dreamer simulator skill score) cannot score it. This is not a physics benchmark and it is not on the same quality scale as Dreamer, Wan or Cosmos. What is measured is the model's own job: encode a context clip, forecast the encoder's latents for the frames that follow, and compare with the latents the encoder produces for the frames that actually followed.

- **No future leakage.** `transformers`' `VJEPA2Model.forward` runs the encoder over the whole clip and applies the masks afterwards, so its "context" tokens have already seen the future. The wrapper instead encodes only the context frames and calls the predictor directly on those tokens (`tests/test_vjepa2.py` checks the forecast against a hand-built context-only run).
- **Task.** 64 frames sampled evenly across a ~10 s clip; the first 32 are the context; the model forecasts the next 16 tubelet-steps (32 frames). The frame spacing is ~0.16 s, while the model was trained at 4 fps (0.25 s), so this is not its training regime; every configuration sees the same input.
- **Data.** 10 public Kinetics clips (`nateraw/kinetics-mini`, the first 2 by name of each of archery, bowling, flying_kite, high_jump, marching). Not redistributed here.
- **Truth and reference.** Truth is the unoptimized fp32 encoder's latents of the full 64-frame clip at the future positions. The reference forecast is the unoptimized fp32 forecast. Both are produced once, before any configuration runs.
- **Timing** is the model only (input prepared beforehand; the CPU resize, 17-54 ms per context in repeated measurements, is excluded and the same for every configuration), warmed up, median of 3 per clip, one process per configuration. `warmup` is the first call (the compile, for compiled configurations).

## Results (mean over 10 clips)

| configuration | latency | speedup | agreement with fp32 forecast (cosine) | peak GPU mem | warm-up |
|---|---|---|---|---|---|
| baseline (fp32) | 688 ms | 1.00 | 1.0000 | 1.65 GB | 1.3 s |
| baseline (repeat) | 689 ms | 1.00 | 1.0000 | 1.65 GB | 1.3 s |
| tf32 | 560 ms | 1.23x | 1.0000 | 1.65 GB | 1.2 s |
| bf16 autocast | 179 ms | 3.85x | 0.9997 | 2.19 GB | 1.0 s |
| fp16 autocast | 177 ms | 3.89x | 1.0000 | 2.19 GB | 1.2 s |
| **bf16 weights** | **159 ms** | **4.34x** | 0.9997 | **0.83 GB** | 1.0 s |
| int8 weight-only (fp32 weights) | 714 ms | 0.96x | 0.9998 | 0.74 GB | 1.1 s |
| int8 dynamic | 1469 ms | 0.47x | 0.9938 | 0.80 GB | 1.8 s |
| bf16 weights + fp8 weight-only | 176 ms | 3.92x | 0.9984 | 0.53 GB | 0.9 s |
| bf16 weights + int8 weight-only | 180 ms | 3.82x | 0.9994 | 0.53 GB | 0.8 s |
| torch.compile (fp32) | 649 ms | 1.06x | 1.0000 | 1.61 GB | 35.6 s |
| **bf16 autocast + torch.compile** | **135 ms** | **5.12x** | 0.9997 | 1.57 GB | 5.5 s* |

\* The compile cost of this row is understated: inductor's on-disk cache was already warm from the `compile` row, which paid 35.6 s. A cold bf16 compile would take longer than 5.5 s.

## How to read it

1. **V-JEPA 2 is compute-bound, so precision is the lever** (3.9-4.3x), the opposite of the launch-bound Dreamer model, where CUDA graphs gave ~9x and precision was slower (`DREAMER_SETUP.md`). Which optimization helps depends on where the time goes; this is why the benchmark profiles before it optimizes.
2. **Quantization on top of fp32 weights does not help.** int8 weight-only is 0.96x and int8 dynamic 0.47x (with the largest fidelity loss, cosine 0.9938). Once the weights are already bf16, int8 and fp8 weight-only land at 3.8-3.9x, about the same speed as bf16 weights alone (4.34x) and slower, in exchange for 0.3 GB less memory. That agrees with the Dreamer and Wan2.1 results.
3. **bf16 weights beat bf16 autocast** (159 vs 179 ms) and use far less memory (0.83 vs 2.19 GB), because autocast keeps casting fp32 weights on every call and caches the casts.
4. **torch.compile is a small win on its own (1.06x) and larger on top of bf16** (5.12x total, 1.33x over bf16 autocast alone). The ViT's matmuls dominate, so fusing the elementwise work around them helps only once the matmuls are fast.
5. **No configuration changed the forecast's quality.** The forecast is a modest improvement on persistence (skill 0.162 +- 0.006: its relative L2 to the true future latents is about 16% lower than repeating the last context step), and that number is identical to three decimals in every row, as is the cosine to truth (0.516, spread across clips +-0.013). The agreement column shows the numerical change each optimization made, and none of them is visible in the quality score.

## Limits

- **Latent fidelity only.** Cosine and relative L2 in V-JEPA 2's own representation space say nothing about pixels or physics. "Optimizations kept the forecast the same" means the latents moved by 0-5.7% in relative L2 depending on the configuration (0 for plain compile, 0.2% for tf32, 2.5% for bf16, 5.7% for bf16 weights + fp8) and by 10% for int8 dynamic, not that anything downstream is unaffected. Whether such a change alters what the representation is used for (planning, classification) was not tested; the Dreamer policy-ranking test is the only experiment in this repo that asks that question.
- **Ten clips from five Kinetics classes, one GPU.** The standard errors above are across clips, and the speedups barely vary between them, but a different GPU or resolution could rank the configurations differently. fp8 needs the hardware support this GPU has (sm_120) and was tested only here.
- **The forecast is only modestly better than persistence** on real video (skill 0.162), and the task (0.16 s spacing, a 16-step horizon) is outside the model's training regime. These results are about optimization effects, not about how good V-JEPA 2 is.
- **The action-conditioned V-JEPA 2-AC** is a separate checkpoint and is not wrapped; `generate` rejects `actions`.
- The latent metrics are not on the PAES scale (no physics term), and PAES is not computed for this model.
