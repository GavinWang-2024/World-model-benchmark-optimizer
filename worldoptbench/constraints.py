"""Constraints on an optimization stack: the `constraints={...}` argument of the outline (section 3.4).

    constraints = Constraints(min_physics_score=0.85, target_speedup=2.0, max_vram_gb=80)
    stack = OptimizationStack(model, modules, constraints=constraints)
    results = run_benchmark(stack.apply(), ...)
    print(stack.check(results).format())          # which constraints the measured run met

and, to *search* under them, `autotune(candidates, evaluate, constraints=constraints)`: it rejects stacks that break
`min_physics_score` or `max_vram_gb`, stops adding modules once `target_speedup` is reached (more speed is not
wanted at the price of more risk), and reports whether the best stack met everything.

The outline sketches a "constraint solver". It cannot be a predictor here: on the Dreamer model predictions about
what helps kept being wrong, so the constraints are checked against *measured* numbers, never estimated.

Definitions, all means over the rollouts of a run unless noted:
  physics   the primary physics score (simulator skill for Dreamer; `None` for text-to-video, where there is none)
  speedup   baseline latency / latency
  vram      the largest GPU memory a rollout needed: max(reserved, peak) over rollouts, in GB (reserved includes
            persistent pools such as CUDA graphs that the per-call peak cannot see)
  latency   seconds per rollout

A constraint whose quantity was not measured (no physics score, no CUDA) is reported as a violation that "could not
be verified", not silently passed.
"""

from __future__ import annotations

import statistics as st
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Constraints:
    min_physics_score: float | None = None
    target_speedup: float | None = None
    max_vram_gb: float | None = None
    max_latency_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("target_speedup", "max_vram_gb", "max_latency_seconds"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_physics_score is not None and not 0.0 <= self.min_physics_score <= 1.0:
            raise ValueError("min_physics_score must be in [0, 1]")


@dataclass
class Measured:
    physics: float | None = None
    speedup: float | None = None
    vram_gb: float | None = None
    latency_seconds: float | None = None


@dataclass
class Violation:
    name: str
    required: float
    actual: float | None
    message: str


@dataclass
class ConstraintReport:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    measured: Measured = field(default_factory=Measured)

    def format(self) -> str:
        m = self.measured

        def show(value: float | None, unit: str = "") -> str:
            return "n/a" if value is None else f"{value:.3f}{unit}"

        head = (
            f"physics {show(m.physics)} | speedup {show(m.speedup, 'x')} | "
            f"vram {show(m.vram_gb, ' GB')} | latency {show(m.latency_seconds, ' s')}"
        )
        if self.ok:
            return f"constraints met ({head})"
        return "constraints NOT met (" + head + "): " + "; ".join(v.message for v in self.violations)


def _get(result: Any, *path: str) -> Any:
    """Reads result.a.b or result['a']['b'] (RunResult objects and loaded JSON rows)."""
    for key in path:
        if result is None:
            return None
        result = result.get(key) if isinstance(result, dict) else getattr(result, key, None)
    return result


def measure_results(results: Sequence[Any]) -> Measured:
    """Aggregates the rollouts of one run (RunResult objects or the dicts from a results JSON)."""
    from worldoptbench.runner import _primary_physics_score  # noqa: PLC0415  (avoids an import cycle at load)

    if not results:
        return Measured()
    physics = [p for r in results if (p := _primary_physics_score(_get(r, "physics"))) is not None]
    speedups = [s for r in results if (s := _get(r, "speed", "speedup")) is not None]
    latencies = [x for r in results if (x := _get(r, "speed", "latency_seconds")) is not None]
    vram = []
    for r in results:
        for key in ("vram_reserved_gb", "vram_peak_gb"):
            value = _get(r, "speed", key)
            if value is not None:
                vram.append(value)
    return Measured(
        physics=st.mean(physics) if physics else None,
        speedup=st.mean(speedups) if speedups else None,
        vram_gb=max(vram) if vram else None,
        latency_seconds=st.mean(latencies) if latencies else None,
    )


def check_constraints(constraints: Constraints, measured: Measured) -> ConstraintReport:
    violations: list[Violation] = []

    def require(name: str, required: float | None, actual: float | None, ok: bool, text: str) -> None:
        if required is None:
            return
        if actual is None:
            violations.append(Violation(name, required, None, f"{name} could not be verified (not measured)"))
        elif not ok:
            violations.append(Violation(name, required, actual, text))

    c, m = constraints, measured
    require("min_physics_score", c.min_physics_score, m.physics,
            m.physics is not None and c.min_physics_score is not None and m.physics >= c.min_physics_score,
            f"physics {m.physics} is below the required {c.min_physics_score}")
    require("target_speedup", c.target_speedup, m.speedup,
            m.speedup is not None and c.target_speedup is not None and m.speedup >= c.target_speedup,
            f"speedup {m.speedup} is below the target {c.target_speedup}")
    require("max_vram_gb", c.max_vram_gb, m.vram_gb,
            m.vram_gb is not None and c.max_vram_gb is not None and m.vram_gb <= c.max_vram_gb,
            f"vram {m.vram_gb} GB exceeds the limit {c.max_vram_gb} GB")
    require("max_latency_seconds", c.max_latency_seconds, m.latency_seconds,
            m.latency_seconds is not None and c.max_latency_seconds is not None and m.latency_seconds <= c.max_latency_seconds,
            f"latency {m.latency_seconds} s exceeds the limit {c.max_latency_seconds} s")
    return ConstraintReport(ok=not violations, violations=violations, measured=measured)
