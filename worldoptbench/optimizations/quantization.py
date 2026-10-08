"""Weight quantization via TorchAO — the first real optimization module, and
the cheapest one: it's architecture-agnostic and needs only the model's
`torch.nn.Module`, not any denoising-loop internals (outline §3.4, §7.6).

Works on any model wrapper that satisfies `models.base.TorchBacked` (has a
`torch_module()` method). Today that's DreamerWorldModel; the Cosmos wrapper
would need the same method pointing at its DiT.

Status: run against TorchAO 0.18.0 on the Dreamer world model (RTX 5070 Ti,
torch 2.11+cu128): the import path (`from torchao.quantization import
quantize_, <Config>`), every config class name, and the `filter_fn` call all
work, and int8/float8 weight-only and int8/float8 dynamic quantized all 13
Linear layers and produced sensible output. `int4_weight_only` fails here with
"Requires mslk >= 1.0.0" (an extra kernel package TorchAO wants for int4) —
install that, or skip int4.

Things to expect when it runs:
  - Only `nn.Linear` layers are touched, never convolutions. For Dreamer
    that's the RSSM and the MLP heads; the conv encoder/decoder stay float32.
  - Small models may get *slower*, not faster: quantization saves memory
    traffic, and a tiny network isn't memory-bound. Reporting that honestly
    (speedup < 1 at no physics gain) is exactly what the benchmark is for.
    (Speed not yet measured — the GPU was busy training when this was checked.)
  - Float8 schemes need CUDA compute capability >= 8.9 and likely bfloat16
    weights; int4 needs `in_features` divisible by its group size. Use
    `min_features` to skip layers a scheme can't handle.
  - Quantization changes numerics, so drift should be re-measured, not assumed
    unchanged — that's the whole point of PAES.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from worldoptbench.models.base import Architecture, TorchBacked, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module

# scheme name -> TorchAO config class name (all documented in the TorchAO inference docs)
SCHEMES: dict[str, str] = {
    "int8_weight_only": "Int8WeightOnlyConfig",
    "int4_weight_only": "Int4WeightOnlyConfig",
    "float8_weight_only": "Float8WeightOnlyConfig",
    "int8_dynamic": "Int8DynamicActivationInt8WeightConfig",
    "float8_dynamic": "Float8DynamicActivationFloat8WeightConfig",
}


@register_module
class QuantizationModule(OptimizationModule):
    name = "quantization"
    requires = ("torch_module",)
    needs_packages = ("torchao",)
    maturity = "measured"
    summary = "TorchAO weight quantization (int8/fp8). Measured: no gain on the launch-bound Dreamer model; on Wan2.1-1.3B (one clip per scheme, so timings only) every scheme is SLOWER than bf16 (int8 weight-only 0.85x, fp8 weight-only 0.73x, fp8 dynamic 0.80x, int8 dynamic 0.23x) without torch.compile, saving about 1.1 GB of peak memory; int4 needs the `mslk` package."
    supported_architectures: tuple[Architecture, ...] = ("diffusion", "autoregressive", "jepa")

    def __init__(self, scheme: str = "int8_weight_only", min_features: int = 0):
        """
        Args:
            scheme: one of `SCHEMES`.
            min_features: skip Linear layers where in_features or out_features
                is below this. 0 quantizes every Linear.
        """
        if scheme not in SCHEMES:
            raise ValueError(f"Unknown scheme {scheme!r}. Choose from {sorted(SCHEMES)}")
        self.scheme = scheme
        self.min_features = min_features
        self.quantized_layers: int | None = None  # set by apply(), for sanity-checking runs

    @property
    def label(self) -> str:
        return f"{self.name}_{self.scheme}"

    def _config(self) -> Any:
        try:
            import torchao.quantization as ao
        except ImportError as e:
            raise ImportError(
                "QuantizationModule needs torchao — install the `ml` extra "
                '(pip install -e ".[ml]") or `pip install torchao`'
            ) from e
        config_name = SCHEMES[self.scheme]
        try:
            return ao.quantize_, getattr(ao, config_name)()
        except AttributeError as e:
            raise ImportError(
                f"torchao.quantization has no {config_name} (or quantize_) — this torchao "
                "version may predate or have renamed it; check its docs"
            ) from e

    def _filter(self) -> Callable[[Any, str], bool]:
        from torch import nn

        def keep(module: Any, fqn: str) -> bool:
            return (
                isinstance(module, nn.Linear)
                and module.in_features >= self.min_features
                and module.out_features >= self.min_features
            )

        return keep

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not isinstance(model, TorchBacked):
            raise TypeError(
                f"{type(model).__name__} doesn't expose torch_module(), so there is no "
                "nn.Module to quantize — add that method to the model wrapper"
            )
        quantize_, config = self._config()
        module = model.torch_module()
        keep = self._filter()

        matched = sum(1 for fqn, m in module.named_modules() if keep(m, fqn))
        if matched == 0:
            # A silent no-op would produce a benchmark row labelled "quantized"
            # for a model that wasn't.
            raise ValueError(
                f"No Linear layers matched (min_features={self.min_features}); "
                "refusing to report an unquantized model as quantized"
            )

        quantize_(module, config, filter_fn=keep)  # mutates in place
        self.quantized_layers = matched
        return model
