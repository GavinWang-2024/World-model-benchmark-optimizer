# Test commands

Run from the repo root. Python 3.11+.

```bash
# one-time setup (dev extra = pytest + ruff; no torch/GPU needed for the test suite)
pip install -e ".[dev]"

# everything (same as CI)
ruff check .
pytest

# one file
pytest tests/test_runner.py
pytest tests/test_optimizations.py
pytest tests/test_dreamer.py     # fake simulator only — no torch or dreamerv3-torch clone needed
pytest tests/test_physics.py
pytest tests/test_library.py          # compatibility, library_report, stack skipping, autotune (scripted)
pytest tests/test_diffusion_modules.py  # tiny random-weight Wan transformer on CPU; needs diffusers, skipped without it
pytest tests/test_wan_video.py         # frame arithmetic, callback chaining, cached reference scenarios (numpy only)
pytest tests/test_worldcache.py         # WorldCache maths on tensors + hooks on a tiny Wan transformer (CPU); needs torch, diffusers
pytest tests/test_blind_metrics.py     # reference-free video statistics and the VisualMetrics integration (numpy only)
pytest tests/test_adacache.py         # AdaCache codebook/motion maths + hooks on a tiny Wan transformer (CPU)
pytest tests/test_perceptual.py       # DINO/CLIP perceptual scoring arithmetic and wiring with fake embedders (no model download)
pytest tests/test_constraints.py      # Constraints, stack.check, autotune under constraints, time to first frame, profile block
pytest tests/test_fvd.py              # Frechet distance (closed-form cases), PCA reduction, video descriptors (numpy/scipy)
pytest tests/test_cost.py             # $ per generated second, cheapest-above-a-floor, cost/quality front
pytest tests/test_utility.py          # rank-agreement measures for the policy-ranking check (scipy)
pytest tests/test_dreamer_policy.py   # policy variants, start states, imagined/true return loops against a tiny fake world model (CPU, torch)
pytest tests/test_diffusion_more.py  # scheduler, uncond_reuse, cross-attention KV cache, VAE options, MagCache + calibration, diffusion defaults (tiny Wan transformer, CPU)
pytest tests/test_latents.py          # latent-only rollouts: the shared imagination helper and generate_latents wiring (CPU, torch)
pytest tests/test_chunked.py          # chunked rollout driver on fake functions, noise-up-front backends, module (CPU, needs torch)
pytest tests/test_serve.py            # worldserve queue, batching, cache and HTTP layer with a fake model (opens a localhost socket)
pytest tests/test_replay_scenarios.py  # replay scenarios from synthetic episode files; numpy only
pytest tests/test_batch2.py            # sparse decode, latent noise, TensorRT decoder/precision wiring, batching checks, scheduling, cache, default stack
pytest tests/test_library_modules.py  # cudnn_benchmark / channels_last / low_rank / VRAM advisor / hygiene
pytest tests/test_tensorrt_gumbel.py  # fake RSSM + fake tensorrt; no GPU needed
pytest tests/test_dreamer_optimizations.py  # precision / tf32 / lean_scan
pytest tests/test_cuda_graphs.py   # fake graph capture; no CUDA needed
pytest tests/test_quantization.py  # layer-filtering tests are skipped unless torch is installed

# one test
pytest tests/test_runner.py::test_run_benchmark_fits_drift_per_prompt_and_penalizes_paes

# verbose / stop at first failure
pytest -v -x

# tests that need torchmetrics are skipped unless the ml extra is installed
pip install -e ".[ml]"
```

On this machine the project venv is `ai\.venv` (see `DREAMER_SETUP.md`), so use
`.\.venv\Scripts\python.exe -m pytest ...` (or activate it with `.\.venv\Scripts\Activate.ps1` first).
The venv already has torch, so the torch-gated tests will run rather than skip (torchao is not installed, but the
quantization tests fake it).
