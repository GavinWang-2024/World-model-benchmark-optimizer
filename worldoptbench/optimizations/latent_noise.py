"""Scale the latent sampling noise: mode (deterministic) or reduced-noise rollouts.

The imagination step samples each latent from a categorical distribution via
Gumbel-max: `argmax(log_probs + noise_scale * Gumbel)`. This module sets
`noise_scale` on the model's imagination backend:

    scale = 1.0   the true distribution (the default; what the sampler is for)
    scale = 0.0   always the most likely class: a deterministic "mode" rollout
    0 < s < 1     less random than the true distribution

It works on the pure-PyTorch Gumbel backend and the TensorRT backend alike (the
noise is an input to both). If the model has no backend yet, the pure-PyTorch
Gumbel one is installed; a backend an earlier module set is reused, with its
`noise_scale` overwritten here. `tensorrt` / `gumbel_sampling` applied *later*
keep whatever scale is already set, so module order doesn't matter.

Read this one carefully: it is not a free speedup, it changes what is being
predicted. A mode rollout is mean-seeking, which one would expect to score better on a
pixel-error fidelity metric than a faithful sample, while no longer representing
the model's uncertainty (every rollout from one context is the same). A PAES gain
from this module would mean "lower pixel error", not "a better world model". It
costs nothing in speed. A first quick check (one scenario, 2 s horizon, 8 seeds per
setting, trained Dreamer walker) found NO detectable difference: skill 0.263 /
0.269 / 0.263 at scales 1.0 / 0.5 / 0.0, against a seed-to-seed spread of 0.05-0.1.
On the sweeps that followed: mode sampling (scale 0) was clearly WORSE on the 515k model's long random-action horizons
(skill 0.248 vs 0.298 with the same noise streams) and within noise on held-out walking (0.237 vs 0.250). No reliable benefit;
the mean-seeking expectation above did not hold. Measured.
"""

from __future__ import annotations

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@register_module
class LatentNoiseModule(OptimizationModule):
    name = "latent_noise"
    requires = ("imagine_backend",)
    maturity = "measured"
    summary = "Scale the latent sampling noise (0 = deterministic mode). Measured: no reliable benefit; mode sampling was worse on long random-action horizons and within noise on held-out walking."
    supported_architectures: tuple[Architecture, ...] = ("autoregressive",)

    def __init__(self, scale: float = 0.0):
        if scale < 0:
            raise ValueError("scale must be >= 0")
        self.scale = scale

    @property
    def label(self) -> str:
        return f"{self.name}_{self.scale:g}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not hasattr(model, "imagine_backend"):
            raise TypeError(
                f"{type(model).__name__} has no `imagine_backend` hook — see models.base.HasImagineBackend"
            )
        backend = model.imagine_backend
        if backend is None:
            from worldoptbench.models.dreamer_step import TorchGumbelImagination  # noqa: PLC0415

            backend = TorchGumbelImagination()
        if not hasattr(backend, "noise_scale"):
            raise TypeError(f"{type(backend).__name__} has no `noise_scale`, so its noise can't be scaled")
        backend.noise_scale = self.scale
        model.imagine_backend = backend
        return model
