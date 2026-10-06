"""OptimizationModule — base class for plug-in optimizations (outline §3.4,
§7.4).

A module takes a WorldModelInterface and returns one (the same object,
mutated, or a wrapper around it). Working at the wrapper level rather than on
a raw diffusers pipeline is deliberate: the wrapper owns pipeline loading
(CosmosPredict loads lazily, and has separate text2world / video2world
pipelines), so only it knows when and where there's a pipeline to hook. Modules
that need to reach into pipeline internals (WorldCache, AdaCache) will need a
way for the model wrapper to expose its loaded pipeline(s) — design that in
Phase 5 once the real WorldCache source is readable, not speculatively now.

Modules hold their own configuration (constructor args), so `apply` only needs
the model.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from worldoptbench.models.base import Architecture, ModelInfo, WorldModelInterface


class OptimizationModule(ABC):
    # Short unique id, used as the registry key and in result labels
    # (e.g. "worldcache", "fp8_quantization").
    name: ClassVar[str]
    # Architectures this module can be applied to. A diffusion-only module like
    # WorldCache sets ("diffusion",); an architecture-agnostic one like
    # quantization lists all of them.
    supported_architectures: ClassVar[tuple[Architecture, ...]]

    # What a model/machine must provide for this module to make sense. The stack
    # skips a module whose requirements aren't met (with the reason) instead of
    # crashing, so a caller can hand it the whole library and get back whatever
    # applies — see `incompatibility` and `library_report`.
    #   requires:       attribute names the model must expose (its opt-in hooks,
    #                   e.g. "tensor_executor", "torch_module", "imagine_backend")
    #   needs_cuda:     a CUDA device must be available
    #   needs_packages: importable packages (checked without importing them)
    requires: ClassVar[tuple[str, ...]] = ()
    needs_cuda: ClassVar[bool] = False
    needs_packages: ClassVar[tuple[str, ...]] = ()
    # "measured": swept on real hardware and its effect is recorded in
    # DREAMER_SETUP.md. "experimental": implemented and unit-tested but not yet
    # measured — don't assume it helps. New modules start experimental.
    maturity: ClassVar[str] = "experimental"
    summary: ClassVar[str] = ""
    # Modules that compete for one slot (e.g. diffusers allows a single cache technique on a
    # transformer at a time) share a group name; the stack keeps the first and skips the rest,
    # saying why, so handing it the whole library (or letting autotune try pairs) can't crash.
    exclusive_group: ClassVar[str | None] = None
    # False for modules that are not optimizations on their own (they only change *how* the rollout runs
    # and depend on another module being applied first), so autotune does not try them as candidates.
    autotune_candidate: ClassVar[bool] = True

    def incompatibility(self, model: WorldModelInterface) -> str | None:
        """Why this module can't be applied to `model` here, or None if it can.

        Checks, in order: architecture, the model's hooks, CUDA, packages. It
        never imports the packages or touches the GPU beyond asking whether one
        exists, so it is cheap and safe to call on every module in the library.
        """
        info = model.get_info()
        if not self.supports(info):
            return (
                f"{self.name} supports {list(self.supported_architectures)}, "
                f"model {info.name} is {info.architecture}"
            )
        missing = [attr for attr in self.requires if not hasattr(model, attr)]
        if missing:
            return f"{type(model).__name__} has no {', '.join(f'`{m}`' for m in missing)} hook (needed by {self.name})"
        if self.needs_cuda and not _cuda_available():
            return f"{self.name} needs a CUDA device; none is available"
        absent = [pkg for pkg in self.needs_packages if importlib.util.find_spec(pkg) is None]
        if absent:
            return f"{self.name} needs the package(s) {', '.join(absent)}, which aren't installed"
        return None

    @property
    def label(self) -> str:
        """What appears in result files (`OptimizationStack.name`). Defaults
        to `name`; override when configuration changes results enough that two
        instances shouldn't share a label (e.g. different quantization schemes).
        """
        return self.name

    def supports(self, info: ModelInfo) -> bool:
        return info.architecture in self.supported_architectures

    @abstractmethod
    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        """Returns the optimized model — either `model` mutated in place or a
        wrapper that delegates to it. The result must still implement
        WorldModelInterface (including `get_info`) so the benchmark runner
        and later modules in the stack can use it unchanged.
        """
        raise NotImplementedError


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


_REGISTRY: dict[str, type[OptimizationModule]] = {}


def register_module(cls: type[OptimizationModule]) -> type[OptimizationModule]:
    """Class decorator: makes `cls` constructible by name from
    `OptimizationStack(modules=["<cls.name>"])`.
    """
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"Optimization module name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_module_class(name: str) -> type[OptimizationModule]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown optimization module {name!r}. Registered: {sorted(_REGISTRY) or 'none'}"
        ) from None


def available_modules() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class ModuleReport:
    name: str
    applicable: bool | None  # None when no model was given to check against
    reason: str | None  # why not, when applicable is False
    maturity: str
    summary: str
    architectures: tuple[str, ...]


def library_report(model: WorldModelInterface | None = None) -> list[ModuleReport]:
    """Every registered module, and — if a model is given — whether it can be
    applied to that model on this machine and why not if it can't. This is the
    "what's in the library for me" query.
    """
    reports = []
    for name in available_modules():
        module = get_module_class(name)()
        reason = module.incompatibility(model) if model is not None else None
        reports.append(
            ModuleReport(
                name=name,
                applicable=None if model is None else reason is None,
                reason=reason,
                maturity=module.maturity,
                summary=module.summary,
                architectures=tuple(module.supported_architectures),
            )
        )
    return reports
