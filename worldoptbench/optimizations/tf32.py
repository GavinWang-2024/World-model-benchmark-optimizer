"""TF32 matmuls: let float32 matrix multiplies use the tensor cores' TF32 mode
(10-bit mantissa instead of 23) on Ampere or newer GPUs.

Process-global, unlike the other modules: PyTorch exposes this as a global
switch (`torch.set_float32_matmul_precision`), so it affects everything in the
process after `apply()`, including other models. Run each benchmark
configuration in its own process (scripts/sweep_dreamer.py does) so settings
can't leak between runs. `restore()` puts the previous values back.

Needs no model hook, so it works on any model. It only affects float32 matmuls,
so it does nothing under autocast bf16/fp16, and like precision changes in
general it can nudge physics scores — measure with PAES.
"""

from __future__ import annotations

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@register_module
class Tf32Module(OptimizationModule):
    name = "tf32"
    needs_cuda = True
    maturity = "measured"
    summary = "TF32 float32 matmuls (process-wide). Measured: no demonstrated effect on the Dreamer model; on Wan2.1-1.3B (32 clips) the output is identical and the speed unchanged (1.00x), since the transformer already runs in bf16."
    supported_architectures: tuple[Architecture, ...] = ("diffusion", "autoregressive", "jepa")

    def __init__(self) -> None:
        self._previous: tuple[str, bool] | None = None

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise RuntimeError("tf32 needs a CUDA device; none is available")
        if torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError("TF32 needs an Ampere (compute capability 8.0) or newer GPU")
        self._previous = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        return model

    def restore(self) -> None:
        """Undo `apply()` (the setting is process-wide)."""
        if self._previous is None:
            return
        import torch  # noqa: PLC0415

        precision, cudnn_tf32 = self._previous
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        self._previous = None
