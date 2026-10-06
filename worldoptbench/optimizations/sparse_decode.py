"""Sparse decoding: decode only every stride-th frame and interpolate the rest.

Decoding is a real share of a Dreamer rollout (~5 of ~13 ms at a 20 s horizon with
TensorRT + CUDA graphs). This decodes the latent features of frames 0, stride,
2*stride, ... plus the last frame, then linearly interpolates the pixels in
between (exact at the decoded frames). It also covers the catalog's "lower fps +
frame interpolation" idea, since both are the same operation here.

Trade-off: interpolated frames are cross-fades, not real predictions, so fast
motion between key frames is blurred. The latent rollout itself is unchanged.
Stride 1 is "off". Careful reading a score for this one: on a quick check (one
scenario, 2 s horizon, trained Dreamer walker) stride 2 and 4 differed from dense
decoding by 1.6 and 3.5 grey levels on average, yet simulator fidelity came out
slightly *higher* (0.364-0.366 vs 0.357), plausibly because blurring is
mean-seeking and a pixel-error metric rewards that, not because the frames are
better. Confirmed on held-out walking episodes at short horizons, where the same decimation LOSES quality against the
same-noise reference: skill -0.014 / -0.047 / -0.079 at strides 2 / 4 / 8. Speed (random-action set, trt + graphs): 9 / 13 / 15%
faster. Stride 2 is a mild trade; larger strides are not worth it. Measured.
"""

from __future__ import annotations

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@register_module
class SparseDecodeModule(OptimizationModule):
    name = "sparse_decode"
    requires = ("decode_stride",)
    maturity = "measured"
    summary = "Decode every k-th frame and interpolate. Measured: 9-15% faster, but costs real quality on held-out walking at stride >= 4 (skill -0.05 at 4, -0.08 at 8); not a default."
    supported_architectures: tuple[Architecture, ...] = ("autoregressive",)

    def __init__(self, stride: int = 4):
        if stride < 2:
            raise ValueError("stride must be >= 2 (1 would decode every frame, i.e. do nothing)")
        self.stride = stride

    @property
    def label(self) -> str:
        return f"{self.name}_{self.stride}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not hasattr(model, "decode_stride"):
            raise TypeError(f"{type(model).__name__} has no `decode_stride` hook, so there is nothing to thin out")
        model.decode_stride = self.stride
        return model
