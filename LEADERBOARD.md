# WorldOptBench leaderboard

Generated 2026-10-04 by `scripts/make_leaderboard.py` from the result files in this repo. Everything was measured on one machine (NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB)), so speedups say how configurations compare *here*; they are not predictions for other hardware. Two small models, not a general result across world models. Every row is a configuration that was actually run; `OPTIMIZATION_CATALOG.md` lists what was not.

## Dreamer (DreamerV3 walker, a real world model with simulator ground truth)

12 held-out walking scenarios at 2 / 5 / 10 / 20 s (`scripts/sweep_dreamer.py`, medians of 3 runs). Skill is simulator fidelity against freezing the last context frame (0 = no better than a frozen scene). The eager baseline's latency varies by about +-12% between runs, so speedups against it are indicative; compare the absolute latencies. Rows that use the TensorRT or Gumbel sampler (`trt`, `tensorrt`, `latent_noise`) draw different random numbers than the baseline, so their skill change is dominated by sampling noise at this many scenarios (about +-0.05): a change of that size is not evidence of lost fidelity (the sampler was checked separately and is statistically equivalent). Rows that keep the noise stream (CUDA graphs with channels_last, cuDNN benchmark, shared pool) are compared tightly, and low-rank factorization and sparse decoding visibly hurt. Drift of the baseline: -0.092 +- 0.009 skill per second.

| configuration | speedup | latency (ms) | skill | skill change | PAES | GPU held (MB) |
|---|---|---|---|---|---|---|
| baseline | 1.00 | 56.1 | 0.299 | +0.000 | 0.274 | 286 |
| cuda_graphs + cudnn_benchmark | 8.55 | 6.1 | 0.299 | -0.000 | 2.109 | 396 |
| cuda_graphs + channels_last | 8.63 | 6.1 | 0.299 | -0.000 | 2.127 | 394 |
| cuda_graphs + share_pool | 8.90 | 6.0 | 0.299 | +0.000 | 2.229 | 286 |
| cuda_graphs + low_rank 0.5 | 9.23 | 5.7 | 0.185 | -0.114 | 1.463 | 440 |
| cuda_graphs + low_rank 0.25 | 10.23 | 5.1 | 0.112 | -0.187 | 1.015 | 422 |
| trt + latent_noise 0.5 + graphs | 11.81 | 4.3 | 0.266 | -0.033 | 2.298 | 394 |
| trt + sparse_decode 2 + graphs | 12.18 | 4.1 | 0.236 | -0.063 | 2.100 | 350 |
| tensorrt + cuda_graphs | 12.31 | 4.1 | 0.250 | -0.049 | 2.280 | 394 |
| trt + latent_noise 0 + graphs | 12.35 | 4.1 | 0.237 | -0.062 | 2.114 | 394 |
| trt + sparse_decode 4 + graphs | 13.11 | 3.9 | 0.203 | -0.096 | 2.038 | 354 |
| trt + sparse_decode 8 + graphs | 13.27 | 3.8 | 0.171 | -0.128 | 1.752 | 352 |
| trt + trt_decoder + graphs | 13.51 | 3.7 | 0.250 | -0.049 | 2.442 | 350 |
| trt + trt_decoder + sparse 4 | 13.71 | 3.7 | 0.203 | -0.096 | 2.132 | 350 |
| trt fp16 + cuda_graphs | 14.67 | 3.4 | 0.251 | -0.048 | 2.594 | 394 |

## Dreamer: does the optimized model still rank policies correctly? (the outline's functional-utility axis)

Does the optimized world model still rank policies like the real simulator and like the unoptimized model? Nine policies per set (variants of the trained actor: `coarse` spans terrible to good play, `fine` is graded action noise so true returns are close), imagined from 24 held-out walking start states over 60 steps (`scripts/policy_ranking_dreamer.py`). Spearman rank correlation; `baseline_reseed` is the unoptimized model with different random draws, the noise floor: nothing can agree with another model better than that. **With nine policies these measures cannot see small losses**: one swapped pair of policies moves Spearman by about 0.02-0.03, so rows within a few hundredths of the reseed's value are not distinguishable from it.

| model | coarse: vs simulator | coarse: vs unoptimized | fine: vs simulator | fine: vs unoptimized | actor's imagined return (coarse) |
|---|---|---|---|---|---|
| baseline | +0.92 |  | +0.95 |  | 112 |
| baseline_reseed | +0.97 | +0.97 | +0.98 | +0.98 | 112 |
| gumbel | +0.92 | +1.00 | +0.95 | +1.00 | 112 |
| tensorrt_fp16 | +0.92 | +1.00 | +0.95 | +1.00 | 113 |
| latent_noise_0.5 | +0.97 | +0.97 | +0.97 | +0.97 | 106 |
| latent_noise_0 | +0.97 | +0.97 | +0.98 | +0.98 | 99 |
| bf16_autocast | +0.92 | +1.00 | +0.98 | +0.98 | 112 |
| int8_weight_only | +0.92 | +1.00 | +0.95 | +1.00 | 113 |
| low_rank_0.5 | +0.93 | +0.98 | +0.98 | +0.98 | 100 |
| low_rank_0.25 | +0.93 | +0.95 | +0.97 | +0.97 | 40 |

Read the last column with the others: `low_rank_0.25` keeps the ranking (about 0.95) while its imagined returns collapse (the actor scores about 40 against 112 unoptimized), so it is still usable to *order* policies and useless for estimating how good one is. The same model lost most of its physics skill (0.11 against 0.30) in the Dreamer table above.

## Wan2.1-1.3B (text-to-video; the diffusion optimizations)

Wan2.1-T2V-1.3B, 32 clips (16 prompts x 2 seeds), 2 s at 192x320, 30 steps. **There is no physics score for text-to-video here**: quality is agreement with the unoptimized video for the same prompt and seed, measured with DINOv2 and CLIP (`worldoptbench/metrics/perceptual.py`). DINO similarity is sameness (1 = identical), not quality: read it with the noise controls (a numerically equivalent change scores about 0.88; the floor, the controls' 5th percentile, is 0.74) and with the CLIP and consistency columns. Pixel style deviation is a coarse screen (reliable only below about 0.1).

| configuration | speedup | DINO similarity | clips below floor | vs noise | CLIP delta | consistency delta | style deviation |
|---|---|---|---|---|---|---|---|
| perturb_1e-2 | 1.00 | 0.876 | 2/32 | control | +0.003 | -0.001 | 0.081 |
| perturb_1e-4 | 1.00 | 0.888 | 1/32 | control | +0.007 | +0.004 | 0.080 |
| vae_tiling | 0.98 | 1.000 | 0/32 | higher | +0.000 | +0.000 | 0.001 |
| cudnn_bench | 1.00 | 1.000 | 0/32 | higher | +0.000 | +0.000 | 0.001 |
| tf32 | 1.00 | 1.000 | 0/32 | higher | +0.000 | +0.000 | 0.000 |
| kv_cache | 1.02 | 1.000 | 0/32 | higher | +0.000 | +0.000 | 0.000 |
| vae_bf16 | 1.04 | 0.999 | 0/32 | higher | +0.000 | +0.000 | 0.013 |
| lossless_kv+vae_bf16 | 1.06 | 0.999 | 0/32 | higher | +0.000 | +0.000 | 0.013 |
| pab_2 | 1.08 | 0.928 | 1/32 | higher | -0.008 | +0.001 | 0.059 |
| ada_slow30_x4 | 1.15 | 0.959 | 1/32 | higher | +0.001 | +0.004 | 0.051 |
| wc_0.02 | 1.21 | 0.949 | 0/32 | higher | +0.001 | +0.001 | 0.045 |
| cfg_trunc_0.6 | 1.22 | 0.968 | 0/32 | higher | -0.003 | +0.004 | 0.040 |
| fbc_0.05 | 1.24 | 0.865 | 4/32 | same | +0.006 | +0.004 | 0.089 |
| uncond_reuse_2 | 1.29 | 0.358 | 31/32 | LOWER | -0.079 | -0.089 | 0.961 |
| taylorseer_3 | 1.35 | 0.657 | 24/32 | LOWER | -0.024 | -0.009 | 0.291 |
| cfg_trunc_0.4 | 1.37 | 0.919 | 1/32 | higher | -0.011 | +0.003 | 0.058 |
| sched_unipc_shift8_s20 | 1.42 | 0.650 | 24/32 | LOWER | +0.008 | -0.010 | 0.388 |
| uncond_reuse_3 | 1.43 | 0.222 | 32/32 | LOWER | -0.122 | -0.072 | 1.359 |
| sched_unipc_shift1.5_s20 | 1.43 | 0.554 | 28/32 | LOWER | -0.032 | +0.001 | 0.480 |
| sched_unipc_o3_s20 | 1.43 | 0.643 | 26/32 | LOWER | -0.012 | -0.013 | 0.339 |
| sched_unipc_shift5_s20 | 1.43 | 0.764 | 10/32 | LOWER | -0.000 | -0.006 | 0.233 |
| steps_20 | 1.43 | 0.633 | 27/32 | LOWER | -0.008 | -0.015 | 0.332 |
| cfg0.4+pab_2 | 1.43 | 0.930 | 0/32 | higher | -0.005 | +0.004 | 0.057 |
| sched_euler_s20 | 1.44 | 0.639 | 23/32 | LOWER | +0.007 | +0.001 | 0.405 |
| sched_dpmpp_s20 | 1.44 | 0.644 | 25/32 | LOWER | -0.002 | -0.010 | 0.341 |
| kv+uncond2+cfg0.6 | 1.48 | 0.384 | 31/32 | LOWER | -0.079 | -0.030 | 1.079 |
| mag_0.06 | 1.49 | 0.885 | 2/32 | same | +0.005 | +0.004 | 0.081 |
| kv+vae_bf16+cfg0.4+pab | 1.54 | 0.930 | 0/32 | higher | -0.005 | +0.003 | 0.062 |
| ada_fast30_guard0.1 | 1.55 | 0.768 | 11/32 | LOWER | +0.006 | +0.002 | 0.197 |
| wc_0.04 | 1.65 | 0.834 | 6/32 | LOWER | -0.005 | +0.003 | 0.126 |
| s20shift5+cfg0.6 | 1.72 | 0.754 | 12/32 | LOWER | -0.005 | +0.001 | 0.242 |
| ada_slow30 | 1.73 | 0.693 | 14/32 | LOWER | -0.017 | -0.009 | 0.318 |
| fbc_0.1 | 1.73 | 0.679 | 22/32 | LOWER | +0.002 | -0.012 | 0.296 |
| mag_0.12 | 1.80 | 0.861 | 4/32 | same | +0.004 | +0.004 | 0.100 |
| sched_unipc_shift5_s15 | 1.82 | 0.688 | 19/32 | LOWER | -0.002 | +0.001 | 0.279 |
| ada_fast30_guard0.15 | 1.86 | 0.652 | 25/32 | LOWER | +0.002 | -0.016 | 0.314 |
| ada_slow30_x0.5 | 1.96 | 0.570 | 27/32 | LOWER | -0.004 | -0.010 | 0.392 |
| s20shift5+cfg0.4+pab | 1.97 | 0.734 | 16/32 | LOWER | -0.010 | -0.006 | 0.245 |
| cfg0.6+wc_0.04 | 1.98 | 0.832 | 6/32 | LOWER | -0.004 | +0.005 | 0.125 |
| mag_0.24 | 2.03 | 0.855 | 4/32 | same | +0.004 | +0.002 | 0.108 |
| mag_1.0 | 2.15 | 0.844 | 4/32 | same | +0.007 | -0.000 | 0.120 |
| mag_0.24_skip5 | 2.15 | 0.833 | 5/32 | LOWER | +0.005 | -0.002 | 0.133 |
| mag_0.5 | 2.15 | 0.844 | 4/32 | same | +0.007 | -0.000 | 0.120 |
| cfg0.6+ada_fast30_guard0.15 | 2.22 | 0.648 | 25/32 | LOWER | -0.003 | -0.012 | 0.318 |
| mag_0.24_ret0.1 | 2.30 | 0.646 | 23/32 | LOWER | -0.010 | -0.008 | 0.319 |
| mag_0.24+cfg0.6 | 2.31 | 0.857 | 3/32 | same | +0.003 | +0.006 | 0.113 |
| wc_0.08 | 2.41 | 0.576 | 29/32 | LOWER | -0.035 | -0.031 | 0.429 |
| mag_0.5_skip5 | 2.47 | 0.818 | 5/32 | LOWER | +0.009 | -0.002 | 0.151 |
| ada_fast30 | 2.51 | 0.454 | 31/32 | LOWER | -0.058 | -0.039 | 0.775 |
| mag_0.24+cfg0.6+kv+vae | 2.54 | 0.856 | 3/32 | same | +0.003 | +0.006 | 0.120 |
| mag_0.5+cfg0.6+kv+vae | 2.63 | 0.847 | 4/32 | same | +0.005 | +0.005 | 0.129 |

## Reproducing a row

```bash
python scripts/sweep_dreamer.py --checkpoint ../dreamerv3-torch/logdir/walker_long --episodes ../dreamerv3-torch/logdir/walker_long/eval_eps --min-return 600
python scripts/sweep_wan.py --standard-set worldoptbench/prompts/wan_set_16.json --out-dir results/wan_perceptual
python scripts/analyze_perceptual.py results/wan_perceptual
```

Details, caveats and the negative results are in `DREAMER_SETUP.md` and `DESIGN_DIFFUSION.md`.
