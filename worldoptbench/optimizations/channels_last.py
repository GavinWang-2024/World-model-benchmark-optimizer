"""channels_last memory format for convolution weights.

Converts the model's 4-D parameters (convolution kernels) to NHWC layout, which
tensor-core convolution kernels generally prefer, avoiding layout transposes
inside cuDNN. Linear layers and other parameters are untouched. Numerics are
identical: it changes memory layout, not values.

Expectation: negligible on the Dreamer model (small 64x64 convolutions, and the
whole rollout is launch-bound); more likely to matter for large convolutional
encoders/decoders such as a video model's VAE. Measured on the Dreamer
model: no gain (23.6 vs 21.0 ms with graphs, inside the 13% noise). Needs only the model's `torch_module()`.
"""

from __future__ import annotations

from worldoptbench.models.base import Architecture, TorchBacked, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@register_module
class ChannelsLastModule(OptimizationModule):
    name = "channels_last"
    requires = ("torch_module",)
    maturity = "measured"
    summary = "NHWC layout for conv weights. Same output; measured: no gain on the Dreamer model (23.6 vs 21.0 ms with graphs, inside the noise)."
    supported_architectures: tuple[Architecture, ...] = ("diffusion", "autoregressive", "jepa")

    def __init__(self) -> None:
        self.converted_parameters: int | None = None  # how many 4-D parameters were converted, for sanity checks

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not isinstance(model, TorchBacked):
            raise TypeError(
                f"{type(model).__name__} doesn't expose torch_module(), so there is nothing to convert"
            )
        import torch

        module = model.torch_module()
        module.to(memory_format=torch.channels_last)  # only 4-D tensors are affected
        self.converted_parameters = sum(1 for p in module.parameters() if p.dim() == 4)
        if self.converted_parameters == 0:
            # A silent no-op would produce a row labelled channels_last for a model that has no convolutions.
            raise ValueError("the model has no 4-D (convolution) parameters; channels_last would do nothing")
        return model
