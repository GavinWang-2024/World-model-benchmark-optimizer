"""cuDNN autotune: let cuDNN time candidate convolution algorithms per input
shape and pick the fastest (`torch.backends.cudnn.benchmark = True`).

Process-global, like tf32 and no_dist_validation: it affects every convolution in
the process, and `restore()` undoes it. The first call at each new shape is slower
(that's where it searches), so a benchmark must warm up every shape first — the
runner's per-horizon warm-up does.

Expectation: on the Dreamer model the convolutions are a small 64x64 encoder and
decoder (decoding is ~5 ms of a ~13 ms rollout), so any gain is bounded by that;
for diffusion or video models with large convolutions it can matter more. Measured on the
Dreamer model: no effect (20.9 vs 21.0 ms with graphs). Combined with CUDA graphs, the search happens during
warm-up/capture and replays reuse its choice.
"""

from __future__ import annotations

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@register_module
class CudnnBenchmarkModule(OptimizationModule):
    name = "cudnn_benchmark"
    needs_cuda = True
    maturity = "measured"
    summary = "cuDNN convolution autotuning (process-wide). Measured: no effect on the Dreamer model (20.9 vs 21.0 ms with graphs), nor on Wan2.1-1.3B (32 clips: identical output, 1.00x)."
    supported_architectures: tuple[Architecture, ...] = ("diffusion", "autoregressive", "jepa")

    def __init__(self) -> None:
        self._previous: bool | None = None

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cudnn_benchmark needs a CUDA device; none is available")
        self._previous = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True
        return model

    def restore(self) -> None:
        """Undo `apply()` (the setting is process-wide)."""
        if self._previous is None:
            return
        import torch

        torch.backends.cudnn.benchmark = self._previous
        self._previous = None
