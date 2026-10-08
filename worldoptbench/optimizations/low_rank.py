"""Low-rank factorization of Linear layers (truncated SVD).

Replaces `Linear(in, out)` with `Linear(in, r) -> Linear(r, out)`, where the two
factors come from the SVD of the original weight truncated to rank r. Cuts the
parameter count and multiply-adds when r is well below min(in, out), at the cost
of approximation error: a lossy compression, so physics must be re-measured
(PAES) rather than assumed.

Choosing r: factoring a (in x out) layer at rank r costs r*(in+out) parameters
against in*out, so it only saves anything when r < in*out/(in+out) — for a
square layer, r must be under half the width. The default `rank_fraction=0.25` of
min(in, out) saves real parameters on every layer it touches; a layer where the
chosen rank wouldn't shrink it is left alone.

Expectation: poor payoff on the Dreamer model, whose layers are small and whose
rollout is launch-bound — splitting one kernel into two adds a launch for each
factored layer. It is here because it's the standard compression baseline and may
pay off on larger models. Layers that aren't plain tensors (e.g. already
quantized by TorchAO) are skipped, so apply this *before* quantization.

Measured on the trained Dreamer walker: rank 0.25 ran ~13% faster (inside the 11% noise; I had predicted slower, but the
weight-reading GEMVs are memory-bound) yet cost heavily in physics on held-out walking episodes: skill 0.299 -> 0.112 at
rank_fraction 0.25 and 0.185 at 0.5, PSNR about -1.8 dB. Not usable on this model without finetuning.
"""

from __future__ import annotations

import math

from worldoptbench.models.base import Architecture, TorchBacked, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@register_module
class LowRankModule(OptimizationModule):
    name = "low_rank"
    requires = ("torch_module",)
    maturity = "measured"
    summary = "Truncated-SVD factorization of Linear layers. Measured: harms physics badly on held-out walking (skill 0.30 -> 0.11 at rank 0.25); do not use on this model."
    supported_architectures: tuple[Architecture, ...] = ("diffusion", "autoregressive", "jepa")

    def __init__(self, rank_fraction: float = 0.25, min_features: int = 64):
        """
        Args:
            rank_fraction: kept rank as a fraction of min(in_features, out_features).
            min_features: leave layers smaller than this (in either dimension) alone.
        """
        if not 0.0 < rank_fraction < 1.0:
            raise ValueError("rank_fraction must be in (0, 1)")
        self.rank_fraction = rank_fraction
        self.min_features = min_features
        self.factored_layers: int | None = None
        self.parameters_before: int | None = None
        self.parameters_after: int | None = None

    @property
    def label(self) -> str:
        return f"{self.name}_{self.rank_fraction:g}"

    def _factor(self, linear):
        import torch
        from torch import nn

        weight = linear.weight.data
        out_features, in_features = weight.shape
        rank = max(1, math.ceil(self.rank_fraction * min(in_features, out_features)))
        if rank * (in_features + out_features) >= in_features * out_features:
            return None  # factoring wouldn't make the layer smaller
        u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        root = s[:rank].sqrt()
        first = nn.Linear(in_features, rank, bias=False).to(weight.device, weight.dtype)
        second = nn.Linear(rank, out_features, bias=linear.bias is not None).to(weight.device, weight.dtype)
        with torch.no_grad():
            first.weight.copy_((root[:, None] * vh[:rank]).to(weight.dtype))
            second.weight.copy_((u[:, :rank] * root[None, :]).to(weight.dtype))
            if linear.bias is not None:
                second.bias.copy_(linear.bias.data)
        first.requires_grad_(False)
        second.requires_grad_(False)
        return nn.Sequential(first, second)

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not isinstance(model, TorchBacked):
            raise TypeError(
                f"{type(model).__name__} doesn't expose torch_module(), so there is nothing to factor"
            )
        import torch
        from torch import nn

        module = model.torch_module()
        self.parameters_before = sum(p.numel() for p in module.parameters())
        factored = 0
        for parent in list(module.modules()):
            for child_name, child in list(parent.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                if min(child.in_features, child.out_features) < self.min_features:
                    continue
                if type(child.weight.data) is not torch.Tensor:  # e.g. a TorchAO-quantized tensor subclass
                    continue
                replacement = self._factor(child)
                if replacement is not None:
                    setattr(parent, child_name, replacement)
                    factored += 1
        self.factored_layers = factored
        self.parameters_after = sum(p.numel() for p in module.parameters())
        if factored == 0:
            # A silent no-op would produce a row labelled low_rank for an unmodified model.
            raise ValueError(
                f"no Linear layer was eligible (min_features={self.min_features}, "
                f"rank_fraction={self.rank_fraction}); refusing to report an unmodified model as factored"
            )
        return model
