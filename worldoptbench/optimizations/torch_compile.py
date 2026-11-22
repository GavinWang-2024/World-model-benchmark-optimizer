"""torch.compile — compile the model's `nn.Module` (or chosen submodules) in place.

Architecture-agnostic, like quantization: it needs only `torch_module()`. Useful where the model is compute-bound
(a ViT such as V-JEPA 2, a diffusion transformer): inductor fuses the elementwise work around the matmuls and
generates Triton kernels. It does the opposite of what a launch-bound model needs, and is a poor fit for the Dreamer
RSSM loop (where CUDA graphs are the lever), so it is not applied there.

`targets` names submodules by attribute path (e.g. `("encoder", "predictor")`) for wrappers that call those
directly instead of the parent's `forward`; compiling only the parent would then compile nothing that runs. With
no targets the whole module is compiled.

Costs to expect and report:
  - The first call after `apply()` pays the compile (tens of seconds on a ViT-L), so benchmarks must warm up
    before timing; the sweep scripts do.
  - Inductor needs Triton and a C compiler (Linux; not available on this project's Windows box). The module does
    not check for them: `torch.compile` raises on first use if they are missing, which is the honest failure.
  - Shapes are specialized: each new input shape recompiles. Fixed-shape benchmarks (a constant context length) hit
    one graph.
  - Numerics change slightly (fused kernels reorder floating-point operations), so fidelity is re-measured, not
    assumed.
"""

from __future__ import annotations

from operator import attrgetter
from typing import Any

from worldoptbench.models.base import Architecture, TorchBacked, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


@register_module
class TorchCompileModule(OptimizationModule):
    name = "torch_compile"
    requires = ("torch_module",)
    maturity = "experimental"
    summary = "torch.compile of the model (or named submodules) in place; compute-bound models only."
    supported_architectures: tuple[Architecture, ...] = ("diffusion", "autoregressive", "jepa")

    def __init__(
        self,
        mode: str | None = "default",
        backend: str = "inductor",
        targets: tuple[str, ...] | None = None,
        fullgraph: bool = False,
    ):
        """
        Args:
            mode: `torch.compile` mode ("default", "reduce-overhead", "max-autotune"); only used by inductor.
            backend: "inductor" (the real thing) or e.g. "eager" / "aot_eager" for wiring tests without a compiler.
            targets: attribute paths of the submodules to compile; None compiles the whole module.
            fullgraph: forbid graph breaks (errors instead of silently falling back to eager for that region).
        """
        self.mode = mode
        self.backend = backend
        self.targets = tuple(targets) if targets else None
        self.fullgraph = fullgraph
        self.compiled: list[str] | None = None  # set by apply(), for sanity-checking runs

    @property
    def label(self) -> str:
        return f"{self.name}_{self.mode}" if self.backend == "inductor" and self.mode else self.name

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not isinstance(model, TorchBacked):
            raise TypeError(
                f"{type(model).__name__} doesn't expose torch_module(), so there is no nn.Module to compile"
            )
        root = model.torch_module()
        kwargs: dict[str, Any] = {"backend": self.backend, "fullgraph": self.fullgraph}
        if self.backend == "inductor" and self.mode:
            kwargs["mode"] = self.mode

        names = self.targets or ("",)
        resolved = []
        for name in names:
            try:
                resolved.append((name, attrgetter(name)(root) if name else root))
            except AttributeError as e:
                # Failing loudly beats reporting a "compiled" row for a model where nothing was.
                raise ValueError(f"{type(root).__name__} has no submodule {name!r} to compile") from e
        for _, module in resolved:
            module.compile(**kwargs)  # nn.Module.compile: in place, the module stays the same object
        self.compiled = [name or "<whole model>" for name, _ in resolved]
        return model
