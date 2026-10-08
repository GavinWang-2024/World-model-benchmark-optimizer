"""CUDA graphs — replay a model's hot loop as a single GPU launch.

Why this and not quantization: the Dreamer world model is launch-bound. Each
generated frame is one RSSM step made of dozens of tiny kernels in a Python
loop, so the GPU sits mostly idle waiting for the CPU to launch the next one
(measured: ~12% GPU utilization, process pinned on CPU). A CUDA graph records
the whole N-step loop once and replays it with one launch, removing that
overhead. Chosen over `torch.compile` because inductor needs Triton plus a C
toolchain, which is fragile on Windows; CUDA graphs need neither.

Mechanism: models that satisfy `models.base.HasTensorExecutor` route their hot,
static-shape tensor function through `model.tensor_executor(fn, *tensors)`.
This module swaps in `CudaGraphExecutor`, which on first sight of a given
(fn, input shapes) warms `fn` up, captures it, and afterwards just copies
inputs into the captured buffers and replays.

Gotchas, all handled here:
  - Capture forbids host syncs. torch.distributions validates arguments with
    `.all()` (a sync), so validation is switched off during capture.
  - Warm-up and capture consume RNG; the RNG state is restored afterwards so
    the first (capturing) call produces the same output as later replays and as
    eager mode for the same seed.
  - Shapes are static: each distinct input shape (e.g. each horizon) gets its
    own captured graph, built lazily on the first call. Benchmark runs should
    warm up every shape first (the runner's warm-up does) or capture time lands
    in the first measured rollout.
  - Output is cloned out of the graph's static buffer, since the next replay
    overwrites it.

Measured on the Dreamer world model (RTX 5070 Ti, median of 3 runs): ~9x faster
than the eager path (21.7 ms mean vs ~210 ms; the eager baseline itself is noisy,
+-34%), with output essentially identical to eager for the same seed. Host-side
overhead is only ~0.4 ms of a 4-5 ms rollout; the time is in the GPU kernel chain,
which the tensorrt module then shortens.

Caveat — memory: each captured graph keeps a persistent private memory pool, so
resident memory goes UP (allocated 124 vs 70 MB, reserved 692 vs 544 MB on that
model). The benchmark's per-call `vram_peak_gb` doesn't see this (the pool is
allocated at capture time, outside the timed call) and reported a misleading
*drop* (0.22 -> 0.13 GB). Don't read that column as a saving for this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldoptbench.models.base import Architecture, HasTensorExecutor, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@dataclass
class _Captured:
    graph: Any
    inputs: list[Any]  # static input buffers the graph reads from
    output: Any  # static output buffer the graph writes to


class CudaGraphExecutor:
    """Callable with the `tensor_executor` signature: `executor(fn, *tensors)`.

    `fn` must be the same object on every call (graphs are cached on its
    identity plus the input shapes/dtypes) and must be capture-safe — see
    `models.base.HasTensorExecutor`. Single-tensor outputs only.
    """

    def __init__(self, warmup_iters: int = 3, share_pool: bool = False):
        self._warmup_iters = warmup_iters
        # One memory pool for all graphs instead of one each. Safe here because
        # nothing a graph allocates outlives its own replay (inputs are static
        # buffers allocated outside capture; the output is cloned immediately).
        # Unmeasured: needs a sweep to confirm it saves memory without changing outputs.
        self._share_pool = share_pool
        self._pool: Any = None
        self._captured: dict[Any, _Captured] = {}

    @property
    def num_graphs(self) -> int:
        return len(self._captured)

    @staticmethod
    def _key(fn: Any, tensors: tuple[Any, ...]) -> Any:
        return (id(fn), tuple((tuple(t.shape), t.dtype, str(t.device)) for t in tensors))

    def __call__(self, fn: Any, *tensors: Any) -> Any:
        key = self._key(fn, tensors)
        entry = self._captured.get(key)
        if entry is None:
            entry = self._captured[key] = self._capture(fn, tensors)
        for buffer, tensor in zip(entry.inputs, tensors):
            buffer.copy_(tensor)
        entry.graph.replay()
        return entry.output.clone()

    def _capture(self, fn: Any, tensors: tuple[Any, ...]) -> _Captured:
        import torch

        inputs = [t.detach().contiguous().clone() for t in tensors]
        device = inputs[0].device
        rng_state = torch.cuda.get_rng_state(device)
        validate_before = torch.distributions.Distribution._validate_args
        torch.distributions.Distribution.set_default_validate_args(False)
        try:
            # Warm-up on a side stream, as PyTorch's CUDA-graph docs require.
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(self._warmup_iters):
                    fn(*inputs)
            torch.cuda.current_stream(device).wait_stream(side)
            torch.cuda.synchronize(device)

            graph = torch.cuda.CUDAGraph()
            if self._share_pool and self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            with torch.no_grad(), torch.cuda.graph(graph, pool=self._pool if self._share_pool else None):
                output = fn(*inputs)
        finally:
            torch.distributions.Distribution.set_default_validate_args(validate_before)
            torch.cuda.set_rng_state(rng_state, device)
        torch.cuda.synchronize(device)
        return _Captured(graph=graph, inputs=inputs, output=output)


@register_module
class CudaGraphsModule(OptimizationModule):
    name = "cuda_graphs"
    requires = ("tensor_executor",)
    needs_cuda = True
    maturity = "measured"
    summary = "Replay the whole rollout as one CUDA graph (~10x on the Dreamer model). share_pool=True: -172 MB held, identical output, latency effect unclear."
    # The mechanism is architecture-agnostic; whether it helps depends on the
    # model opting in via `tensor_executor` (and being launch-bound).
    supported_architectures: tuple[Architecture, ...] = ("diffusion", "autoregressive", "jepa")

    def __init__(self, warmup_iters: int = 3, share_pool: bool = False):
        self.warmup_iters = warmup_iters
        self.share_pool = share_pool

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not isinstance(model, HasTensorExecutor):
            raise TypeError(
                f"{type(model).__name__} doesn't expose a `tensor_executor`, so there is "
                "nothing for CUDA graphs to wrap — see models.base.HasTensorExecutor"
            )
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cuda_graphs needs a CUDA device; none is available")
        model.tensor_executor = CudaGraphExecutor(self.warmup_iters, self.share_pool)
        return model
