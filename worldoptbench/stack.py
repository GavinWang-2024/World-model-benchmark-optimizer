"""OptimizationStack — filters optimization modules by model architecture and
chains the compatible ones (outline §3.4, §7.4).

    stack = OptimizationStack(model, ["worldcache", "fp8_quantization"])
    optimized = stack.apply()
    run_benchmark(optimized, optimization=stack.name, baseline_path="baseline.json")

Constraints (`worldoptbench.constraints.Constraints`: min_physics_score, target_speedup, max_vram_gb, max_latency_seconds)
are checked against a *measured* run with `stack.check(results)`, and searched under by `autotune(..., constraints=...)`;
nothing is predicted (see constraints.py for why).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from worldoptbench.constraints import ConstraintReport, Constraints, check_constraints, measure_results
from worldoptbench.models.base import WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, get_module_class
from worldoptbench.runner import BASELINE_OPTIMIZATION


@dataclass
class SkippedModule:
    name: str
    reason: str


class OptimizationStack:
    def __init__(
        self,
        model: WorldModelInterface,
        modules: Sequence[str | OptimizationModule] = (),
        module_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        constraints: Constraints | None = None,
    ):
        """
        Args:
            model: the model to optimize.
            modules: registered module names, or already-constructed module
                instances (use instances to pass constructor config directly).
                Applied in the order given. Unknown names raise KeyError —
                a typo shouldn't silently turn into an unoptimized run. A
                module that doesn't fit the model (wrong architecture, missing
                hook, no CUDA, missing package) is skipped and recorded in
                `.skipped` with the reason, so passing the whole library is safe.
            module_kwargs: constructor kwargs for modules given by name,
                keyed by module name. Ignored for instances.
            constraints: optional `Constraints` this stack is meant to satisfy; checked after a run
                with `check(results)`. They do not change which modules are applied.
        """
        self._model = model
        self.constraints = constraints
        self._applied_once = False

        self.modules: list[OptimizationModule] = []
        self.skipped: list[SkippedModule] = []
        taken: dict[str, str] = {}  # exclusive group -> name of the module holding it
        for entry in modules:
            module = (
                entry
                if isinstance(entry, OptimizationModule)
                else get_module_class(entry)(**(module_kwargs or {}).get(entry, {}))
            )
            reason = module.incompatibility(model)
            group = module.exclusive_group
            if reason is None and group is not None and group in taken:
                reason = f"{module.name} conflicts with {taken[group]}: both use the {group!r} slot, which holds one at a time"
            if reason is None:
                self.modules.append(module)
                if group is not None:
                    taken[group] = module.name
            else:
                self.skipped.append(SkippedModule(name=module.name, reason=reason))

    @property
    def name(self) -> str:
        """Label for result files, e.g. "worldcache+fp8_quantization", or
        "baseline" if no module applies. Matches `run_benchmark(optimization=...)`.
        """
        return "+".join(m.label for m in self.modules) or BASELINE_OPTIMIZATION

    def apply(self) -> WorldModelInterface:
        """Chains the compatible modules over the model and returns the result.

        Modules may mutate the model in place, so this can only be called
        once per stack — build a fresh model and stack for another run
        rather than re-applying.
        """
        if self._applied_once:
            raise RuntimeError(
                "OptimizationStack.apply() already called; modules may have mutated the "
                "model in place — create a new model and stack instead"
            )
        self._applied_once = True

        model = self._model
        for module in self.modules:
            model = module.apply(model)
        return model

    def check(self, results: Sequence[Any]) -> ConstraintReport:
        """Whether a measured run (the RunResults, or rows loaded from a results JSON) met this stack's
        constraints. Raises if the stack was built without any."""
        if self.constraints is None:
            raise ValueError("this stack has no constraints to check; pass constraints=Constraints(...)")
        return check_constraints(self.constraints, measure_results(results))

    def restore(self) -> None:
        """Undo process-global side effects (e.g. TF32, distribution validation)
        for every applied module that defines `restore()`. Modules that only
        change the model need nothing. Call this when running several stacks in
        one process; separate processes need no cleanup.
        """
        for module in self.modules:
            restore = getattr(module, "restore", None)
            if callable(restore):
                restore()
