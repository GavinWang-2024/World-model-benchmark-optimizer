"""Turn off torch.distributions argument validation.

By default every `Distribution` the model builds checks its parameters with
`.all()` reductions, each of which forces a GPU->CPU sync. The repo's RSSM
builds several distributions per imagination step, so the eager loop is full of
them (6 reduction kernels and 3 device-to-host copies per step in a profile).
Disabling validation changes nothing numerically — outputs differ from the
validated run only by the same rare +-1/255 jitter two identical runs show —
and measured on the trained Dreamer walker it speeds the eager path up
**1.43x** (431 -> 302 ms at a 20 s horizon).

It does nothing for CUDA-graph runs: the graph executor already disables
validation while capturing (a host sync can't be captured), so graph replays
never had those kernels. Process-global, like tf32: `restore()` undoes it.
"""

from __future__ import annotations

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@register_module
class NoDistValidationModule(OptimizationModule):
    name = "no_dist_validation"
    maturity = "measured"
    summary = "Disable torch.distributions validation: ~1.4x on the eager path, no effect with graphs."
    supported_architectures: tuple[Architecture, ...] = ("diffusion", "autoregressive", "jepa")

    def __init__(self) -> None:
        self._previous: bool | None = None

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        import torch

        self._previous = torch.distributions.Distribution._validate_args
        torch.distributions.Distribution.set_default_validate_args(False)
        return model

    def restore(self) -> None:
        if self._previous is None:
            return
        import torch

        torch.distributions.Distribution.set_default_validate_args(self._previous)
        self._previous = None
