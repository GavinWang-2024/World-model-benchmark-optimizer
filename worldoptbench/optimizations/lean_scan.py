"""Lean scan: a cheaper imagination loop for recurrent latent models.

The Dreamer repo imagines forward with `tools.static_scan`, which after every
step re-`torch.cat`s each output onto a growing buffer (so an N-step rollout
copies O(N^2) data) and carries statistics (mean/std/logit) that nothing
downstream reads. This switches the model to a loop that runs the same
steps but keeps only the two tensors the decoder needs and stacks them once.
Numerics are identical — it removes work, it doesn't approximate.

It's an algorithmic win, separate from the launch-overhead win that CUDA graphs
give: the copies become real GPU time once launch overhead is gone, so the two
are worth measuring together. Dreamer-specific mechanism (the model opts in via
a `lean_scan` flag, `HasLeanScan` below), so it only declares the recurrent
architecture.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@runtime_checkable
class HasLeanScan(Protocol):
    """Model with a switchable imagination loop (`lean_scan: bool`)."""

    lean_scan: Any


@register_module
class LeanScanModule(OptimizationModule):
    name = "lean_scan"
    requires = ("lean_scan",)
    maturity = "measured"
    summary = "Collect-and-stack imagination loop. Measured: no demonstrated effect."
    supported_architectures: tuple[Architecture, ...] = ("autoregressive",)

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not isinstance(model, HasLeanScan):
            raise TypeError(
                f"{type(model).__name__} has no `lean_scan` switch — see "
                "worldoptbench.optimizations.lean_scan.HasLeanScan"
            )
        model.lean_scan = True
        return model
