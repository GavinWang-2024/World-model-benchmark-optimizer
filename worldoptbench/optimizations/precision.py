"""Mixed precision: run the model's forward under `torch.autocast`.

Matmuls and convolutions run in bfloat16/float16 while numerically sensitive
ops (norms, softmax, reductions) stay float32 — which is why this is done with
autocast rather than casting the weights, which could break the RSSM's
distribution sampling. Needs the model to opt in via `autocast_dtype`
(`models.base.HasAutocast`).

Applies the same way to any architecture, but whether it *helps* depends on the
model: on a small launch-bound network the arithmetic isn't the bottleneck, so
expect little gain until something like CUDA graphs removes the launch
overhead. Precision loss shows up as physics/visual drift, so check PAES, not
just speed. float16 has a narrow range and can overflow; bfloat16 is the safer
default on hardware that supports it.

Must be applied before the first `generate()`: a CUDA graph captured earlier
keeps whatever dtype was active at capture time.
"""

from __future__ import annotations

from worldoptbench.models.base import Architecture, HasAutocast, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module

DTYPES = ("bfloat16", "float16")


@register_module
class PrecisionModule(OptimizationModule):
    name = "precision"
    requires = ("autocast_dtype",)
    maturity = "measured"
    summary = "bf16/fp16 autocast. Measured: slower with graphs on the Dreamer model, ~100 MB less memory."
    supported_architectures: tuple[Architecture, ...] = ("diffusion", "autoregressive", "jepa")

    def __init__(self, dtype: str = "bfloat16"):
        if dtype not in DTYPES:
            raise ValueError(f"Unknown dtype {dtype!r}. Choose from {list(DTYPES)}")
        self.dtype = dtype

    @property
    def label(self) -> str:
        return f"{self.name}_{self.dtype}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not isinstance(model, HasAutocast):
            raise TypeError(
                f"{type(model).__name__} doesn't expose `autocast_dtype`, so there is nothing "
                "to switch to mixed precision — see models.base.HasAutocast"
            )
        import torch  # noqa: PLC0415

        if self.dtype == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU doesn't support bfloat16; use dtype='float16'")
        model.autocast_dtype = getattr(torch, self.dtype)
        return model
