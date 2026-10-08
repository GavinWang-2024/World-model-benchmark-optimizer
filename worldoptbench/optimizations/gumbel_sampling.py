"""Gumbel-max sampling, in pure PyTorch: the imagination loop rewritten with a
lean exportable step and Gumbel noise (see models/dreamer_step.py).

Two reasons it exists:
  1. It is itself a modest speedup over the repo's loop: no distribution
     objects, no argument validation, far fewer kernels per step.
  2. It is the exact reference for the TensorRT backend — same noise, same
     math, no TensorRT — so a TensorRT result can be checked against it
     rather than only against the repo's differently-seeded multinomial path.

Statistically equivalent to the repo's sampling (Gumbel-max draws from the same
categorical distribution) but a different random stream, so individual
rollouts differ from the baseline while their average behavior doesn't; use many
rollouts when judging physics changes against the baseline.
"""

from __future__ import annotations

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@register_module
class GumbelSamplingModule(OptimizationModule):
    name = "gumbel_sampling"
    requires = ("imagine_backend",)
    maturity = "measured"
    summary = "Pure-PyTorch Gumbel-max RSSM step: ~1.6x eager, ~1.2x with graphs; reference for tensorrt."
    supported_architectures: tuple[Architecture, ...] = ("autoregressive",)

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not hasattr(model, "imagine_backend"):
            raise TypeError(
                f"{type(model).__name__} has no `imagine_backend` hook — see "
                "models.base.HasImagineBackend"
            )
        from worldoptbench.models.dreamer_step import TorchGumbelImagination

        backend = TorchGumbelImagination()
        # Keep a noise scale a previous module (latent_noise) already set.
        backend.noise_scale = getattr(getattr(model, "imagine_backend", None), "noise_scale", 1.0)
        model.imagine_backend = backend
        return model
