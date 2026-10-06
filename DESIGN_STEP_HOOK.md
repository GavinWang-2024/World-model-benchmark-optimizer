# Design: per-step hooks and chunked graphs

Status: **step 1 (chunked execution, no hooks) is implemented and measured** (see "Step 1 results" at the end); steps 2-4 (tensor hooks, chunk hooks, drift correction) are not started. The rest of this document is the original design, written 2026-10-03 while the GPU was busy training.

## Why

About a dozen entries in `OPTIMIZATION_CATALOG.md` need code to run *inside the rollout, between steps*: uncertainty-gated early stop and recompute (C6, M4), re-anchoring to real observations (M3), sparse decoding (C3), mixed-fidelity schedules (M5), mode/noise-scaled sampling (M7, M8), quantization-induced drift correction (M2, the outline's front-runner novel contribution), and the diffusion-side equivalents later. Today the whole rollout is one opaque function that either runs eagerly or is replayed as a single CUDA graph.

## The constraint that shapes everything

A CUDA graph records GPU kernels once and replays them. **Python only runs at capture time.** Consequences, all hit while building what exists:

- A hook that is a *tensor function* (no host sync, no Python branching on tensor values) can live inside a captured graph. It is baked in, and runs at full speed on every replay.
- A hook that needs a **host decision** (stop now? re-anchor now? recompute?) must read a tensor on the CPU, which is a sync. That is illegal inside capture (this is why the repo's `obs_step`, which does `if torch.sum(is_first) == ...`, had to be rewritten), and it can't be baked in, because the decision changes per rollout.
- Running every step eagerly so any hook can be arbitrary Python costs the whole 9x (back to ~210 ms from ~22 ms).

So hooks come in two kinds, and the design is to support both without giving up graphs.

## Proposed design

**Two hook levels**

1. **Step hooks: pure tensor functions**, `hook(t, state, ctx) -> state`, applied inside the imagination loop and captured with it. For anything computable on-device: noise scaling, mode sampling (M7/M8), a precision schedule that is a function of `t` (M5), writing a per-step uncertainty value into a buffer. No host reads allowed; the executor enforces this by running the hook once under a check that raises on sync during capture (it already fails loudly today, which is what we want).
2. **Chunk hooks: host-side Python between graph replays**, `hook(chunk_index, state_summary) -> Action`. The rollout is split into chunks of K steps. Each chunk is one captured graph; between chunks the host may sync, inspect a summary tensor, and return `CONTINUE`, `STOP`, `REANCHOR(posterior_state)` or `RECOMPUTE_CHUNK(precision)`. Early stop, re-anchoring and uncertainty-gated recompute (C6, M3, M4) are all chunk-level decisions.

**Chunked graphs also fix things that are independently wrong today**

- One small graph per chunk size is reused for *any* horizon (replay ceil(N/K) times), instead of one graph per horizon. That cuts capture time and the persistent memory graphs hold (+128 MB for 4 horizons on the Dreamer model; catalog A6, A11).
- It enables streaming output (catalog L3): frames are available after each chunk.

**What changes in the code**

- `_imagine_decode` becomes: observe -> loop over chunks (executor-wrapped `chunk_fn(stoch, deter, action_chunk) -> (stoch', deter', features_chunk)`) -> decode. The decode stays one batched call at the end, or per chunk when streaming. With K = N and no hooks it must reduce exactly to today's single graph.
- `imagine_backend` (TensorRT / Gumbel) needs a chunk form: `backend(wm, stoch, deter, actions_chunk)`. For TensorRT that is K enqueues per chunk instead of N, a trivial change. The engine is already one-step.
- A new `rollout_hooks` attribute on the model (a `HasRolloutHooks` protocol, like the existing `tensor_executor`, `autocast_dtype`, `imagine_backend`), set by modules before the first `generate()`.
- `run_benchmark` needs nothing: hooks live behind `generate()`. Hook-induced early stops return fewer frames, so the metrics need to handle a short rollout (score only the frames produced; record `stopped_at`). That is a real interface change and should be decided explicitly, not discovered.

**Chunk size K.** Replaying a graph costs on the order of tens of microseconds of launch plus a host sync at each boundary (~0.1 ms). At 20 steps per second a 20 s rollout with K = 25 is 16 chunks, so the overhead should be well under 2 ms against ~13 ms of TensorRT-in-graph work. K trades that overhead against how quickly a hook can react. Candidate default: 20 steps (1 s of simulated time). Measure it, don't assume it.

## Risks and unknowns (to settle on the GPU)

1. **TensorRT inside many small graphs.** The prototype captured 400 TensorRT enqueues in one graph. Nothing yet shows chunked replay keeps the 28 us/step. It may, but TensorRT's execution context could add per-graph synchronization.
2. **The cost of chunk boundaries may erase the gain for short rollouts.** At a 1-2 s horizon the whole rollout is 4-5 ms, so even 0.1 ms per boundary matters.
3. **Equivalence.** A rollout with no-op hooks and chunking must match the unchunked rollout (same seed): bit-near-identical on the eager/PyTorch path, and within the already-measured TensorRT-vs-PyTorch agreement on the TensorRT path. RNG streams must not depend on K: draw all noise up front for the whole rollout, then slice per chunk. This is the easiest thing to get subtly wrong.
4. **State dependence across the executor cache.** Graphs are cached by function identity and shapes; chunk functions must be built once per model, not per call.

## Order of work, once the GPU is free

1. Chunked execution with **no hooks**, TensorRT and PyTorch backends; prove equivalence against the current whole-rollout graphs and measure the boundary overhead for K in {10, 20, 50, 100}. If overhead is unacceptable, stop here and keep whole-rollout graphs for the no-hook case.
2. Step hooks (tensor-level): mode sampling and noise scaling (M7, M8) are the easiest first customers and give an immediate physics-vs-fidelity result.
3. Chunk hooks: early stop and re-anchoring, including the short-rollout metric handling.
4. Only then M2/M4 (quantization-induced drift correction, uncertainty-gated recompute), which need a better-trained model for the drift signal to be real.

## Not in scope

Diffusion denoise-loop hooks (the `DENOISE` hook in the catalog) are a separate design: they need a diffusion model to exist first, and a denoising step has different state than an RSSM step.


## Step 1 results (implemented 2026-10-04): chunked execution with no hooks

Code: `worldoptbench/models/chunked.py` (`run_chunked`, model-agnostic), `DreamerWorldModel.chunk_steps` and its observe / chunk / decode / noise functions, the `chunked_rollout` module (`optimizations/chunked.py`), `draw_noise()` and a `noise=` argument on the Gumbel and TensorRT backends, `scripts/bench_chunked.py`, `tests/test_chunked.py`. The latent state travels packed in one float32 tensor (executors pass single tensors); the last chunk is padded to the common size and trimmed; **all the noise is drawn once, after the observe step and before any chunk**, which is the order the whole-rollout function consumes the random stream in. That only works for imagination backends with explicit noise, so chunking requires `gumbel_sampling` or `tensorrt`; the repo's own sampler (noise drawn inside each step) is refused, as are `decode_stride > 1` and autocast.

Measured on the 515k-step walker, TensorRT fp16 + CUDA graphs, median of 15 calls (ms), RTX 5070 Ti laptop, GPU otherwise idle:

| configuration | 1 s | 2 s | 5 s | 10 s | 20 s | graphs | held MB | max pixel diff | differing pixels |
|---|---|---|---|---|---|---|---|---|---|
| whole rollout (one graph per horizon) | 2.7 | 3.3 | 5.6 | 9.4 | 18.3 | 5 | 126 | | |
| chunked K=10 | 2.9 | 4.0 | 7.4 | 13.6 | 26.1 | 3 | 215 | 1 | 0.46% |
| chunked K=20 | 2.8 | 3.8 | 6.8 | 11.7 | 22.4 | 3 | 304 | 1 | 0.39% |
| chunked K=50 | 3.9 | 3.8 | 5.9 | 10.0 | 19.2 | 3 | 394 | 1 | 0.41% |
| chunked K=100 | 5.6 | 5.7 | 5.8 | 9.8 | 18.8 | 3 | 485 | 1 | 0.41% |

(The pure-PyTorch Gumbel backend gave the same picture: K=20 vs whole, 1 s 3.5 vs 3.3 ms, 5 s 10.4 vs 9.2 ms, max pixel difference 1.)

What it answers, against the risks listed above:

- **Equivalence (risk 3): yes.** The frames match the whole-rollout graph to within one grey level at every chunk size, with 0.4% of pixels differing by that one level (the same size as run-to-run GPU rounding jitter). The random stream does not depend on K.
- **Boundary cost (risks 1 and 2): about 0.2 ms per chunk**, from the extra executor calls (input copies, replay, output clone) for the chunk and its decode. At 20 s that is +3% (K=100), +5% (K=50), +22% (K=20) and +43% (K=10) on top of the whole-rollout time. TensorRT inside small graphs did not break the per-step speed.
- **Short horizons pay for padding**: with K=100 a 1 s rollout (20 steps) computes 100, 5.6 ms against 2.7 ms. A remainder-sized graph would avoid this at the price of one more graph per distinct remainder.
- **Graph count is constant (3: observe, chunk, decode) as the design hoped, but held memory was not lower**: 215-485 MB against 126 MB for the five whole-rollout graphs, growing with K. Not investigated; the "less persistent graph memory" benefit is not supported in these measurements.
- **Verdict (the design's own stop rule):** chunking is not a speedup and it costs memory here, so keep whole-rollout graphs for the no-hook case. Use `chunked_rollout` only when the chunk boundaries are wanted (hooks, streaming), with K of 50 or more if the extra latency matters (K=50 is 2.5 s of simulated time between host decisions). `autotune` does not try it as a candidate (`autotune_candidate = False`).

Still open (the design's steps 2-4): tensor-level step hooks, chunk hooks (early stop, re-anchoring, and the short-rollout metric handling they need), and anything that depends on a drift or uncertainty signal.
