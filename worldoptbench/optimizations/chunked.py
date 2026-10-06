"""Chunked rollout: run the imagination loop as fixed-size chunks (see models/chunked.py).

On its own this is not a speedup. A chunk is a small tensor function, so a CUDA graph captured for it is
reused for any horizon (fewer graphs, less persistent graph memory, shorter warm-up) and the host regains
control between chunks, which later hooks (early stop, re-anchoring, streaming) build on; the price is one
executor call per chunk instead of one per rollout. `scripts/bench_chunked.py` measures that price.

Needs an imagination backend that takes pre-drawn noise: apply `gumbel_sampling` or `tensorrt` first.
"""

from __future__ import annotations

from typing import Any

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@register_module
class ChunkedRolloutModule(OptimizationModule):
    name = "chunked_rollout"
    requires = ("chunk_steps",)
    maturity = "measured"
    autotune_candidate = False  # a structure change that needs a backend applied first, not a speedup
    supported_architectures: tuple[Architecture, ...] = ("autoregressive",)
    summary = (
        "Run the rollout as fixed-size chunks so one captured graph serves any horizon (needs gumbel_sampling or "
        "tensorrt). Measured with tensorrt fp16 + graphs: output matches the whole-rollout graph to 1 grey level; "
        "NOT a speedup, about 0.2 ms per chunk boundary (+3-5% at chunk_steps 50-100 on long rollouts, +25-45% at "
        "10-20), and large chunks waste work on short horizons; graph memory was not reduced. Use it for the hooks "
        "it enables, not for speed."
    )

    def __init__(self, chunk_steps: int = 20):
        if chunk_steps < 1:
            raise ValueError("chunk_steps must be >= 1")
        self.chunk_steps = chunk_steps
        self._previous: Any = None
        self._model: Any = None

    @property
    def label(self) -> str:
        return f"{self.name}_{self.chunk_steps}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        backend = getattr(model, "imagine_backend", None)
        if backend is not None and not hasattr(backend, "draw_noise"):
            raise RuntimeError(
                f"{type(backend).__name__} cannot take pre-drawn noise, which chunked rollouts need "
                "(use gumbel_sampling or tensorrt)"
            )
        self._model, self._previous = model, model.chunk_steps
        model.chunk_steps = self.chunk_steps
        return model

    def restore(self) -> None:
        if self._model is not None:
            self._model.chunk_steps = self._previous
