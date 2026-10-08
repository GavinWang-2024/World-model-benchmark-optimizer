"""Let the library choose: greedy search over optimization modules.

Given a way to build a fresh model and a candidate set of modules, `autotune`
starts from no optimization and repeatedly adds whichever single module most
improves PAES, stopping when nothing helps by more than a margin. That is the
"constraint solver" the outline (section 3.4) sketches, in its simplest honest
form: it measures instead of predicting, because on the Dreamer model the
predictions kept being wrong (quantization and mixed precision *hurt*; only
CUDA graphs and TensorRT helped).

Why greedy and not exhaustive: n modules is 2^n stacks, each a full benchmark.
Greedy forward selection costs about n^2/2 evaluations at worst and finds the
stack by construction rather than testing every one. It can miss pairs that only
help together; `max_modules` and `min_gain` keep it from wandering.

Two guards against fooling yourself, both learned on real runs:
  - `min_gain`: a module must beat the current best by this *relative* margin.
    Single benchmark runs on the eager baseline varied +-34%, so a few percent
    is noise; the default is deliberately conservative.
  - `max_physics_drop`: a stack that speeds things up by hurting physics is
    rejected outright, whatever its PAES.

`autotune` itself is a pure search over an `evaluate` callable, so it can be
tested without a GPU; `evaluate_with_runner` builds the real one from a model
factory and `run_benchmark`.

Known limitation (found on its first real run): `max_physics_drop` compares a
candidate's physics score against the baseline's, but for modules that change the
random stream (the Gumbel / TensorRT sampler, latent noise) that difference is
dominated by sampling noise. On 12 scenarios the noise wanders by ~0.05, so a
faithful TensorRT stack scored 0.251 against the baseline's 0.299 and would have
been rejected at the default 0.02. Repeating the benchmark doesn't expose this (the
seeds are fixed, so the physics score is identical run to run); only independent
seeds do. Until the search estimates that noise itself, set `max_physics_drop`
to ~2 standard errors for the scenario count you use (~0.06 for 12 scenarios).
Modules that keep the noise stream and change only weights or the decoder are
compared tightly, so the guard is sharp for those.
"""

from __future__ import annotations

import statistics as st
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from worldoptbench.optimizations.base import available_modules


@dataclass
class Evaluation:
    """What one candidate stack measured: higher `paes` is better."""

    paes: float
    physics: float  # mean physics score; compared against the baseline's to catch quality loss
    latency_seconds: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    speedup: float | None = None  # mean speedup against the baseline, for target_speedup
    vram_gb: float | None = None  # largest GPU memory a rollout needed (reserved or peak), for max_vram_gb


@dataclass
class TuneStep:
    modules: list[str]
    evaluation: Evaluation
    accepted: bool
    note: str = ""


@dataclass
class TuneResult:
    best_modules: list[str]
    best: Evaluation
    baseline: Evaluation
    trace: list[TuneStep]
    report: Any = None  # a constraints.ConstraintReport for the best stack, when constraints were given


Evaluate = Callable[[Sequence[str]], Evaluation]


def autotune(
    candidates: Sequence[str],
    evaluate: Evaluate,
    *,
    min_gain: float = 0.10,
    max_physics_drop: float = 0.02,
    min_physics: float | None = None,
    max_modules: int = 6,
    constraints: Any = None,
) -> TuneResult:
    """Greedy forward selection of `candidates`.

    Args:
        candidates: module names to consider.
        evaluate: maps a list of module names to an `Evaluation` (the empty
            list is the baseline). Must build a fresh model each call, because
            modules mutate models.
        min_gain: accept a module only if the new PAES exceeds the current best
            by this fraction (0.10 = 10%).
        max_physics_drop: reject any stack whose mean physics score is more than
            this far below the baseline's.
        min_physics: optional absolute floor on mean physics score.
        max_modules: stop after this many modules.
        constraints: optional `worldoptbench.constraints.Constraints`. `min_physics_score` raises the physics
            floor and `max_vram_gb` rejects stacks that need more GPU memory (when the evaluation measured it);
            reaching `target_speedup` stops the search (more speed is not wanted at more risk). The best stack's
            `report` says whether everything was met; if the search ends short, it says which constraint failed.

    Returns:
        The best stack found, its evaluation, the baseline's, and every step
        tried (accepted or not), so the choice can be audited.
    """
    baseline = evaluate([])
    best_modules: list[str] = []
    best = baseline
    trace: list[TuneStep] = []
    remaining = list(dict.fromkeys(candidates))  # de-duplicated, order kept

    floor = max([v for v in (min_physics, getattr(constraints, "min_physics_score", None)) if v is not None], default=None)
    vram_limit = getattr(constraints, "max_vram_gb", None)
    target = getattr(constraints, "target_speedup", None)

    def acceptable(ev: Evaluation) -> str | None:
        if vram_limit is not None and ev.vram_gb is not None and ev.vram_gb > vram_limit:
            return f"vram {ev.vram_gb:.2f} GB exceeds the limit {vram_limit:.2f} GB"
        if baseline.physics - ev.physics > max_physics_drop:
            return f"physics {ev.physics:.3f} is {baseline.physics - ev.physics:.3f} below baseline {baseline.physics:.3f}"
        if floor is not None and ev.physics < floor:
            return f"physics {ev.physics:.3f} is below the floor {floor:.3f}"
        return None

    while remaining and len(best_modules) < max_modules:
        if target is not None and best.speedup is not None and best.speedup >= target:
            break  # the speedup target is met; stop rather than chase more at the cost of fidelity
        round_best: tuple[str, Evaluation] | None = None
        for name in remaining:
            stack = [*best_modules, name]
            ev = evaluate(stack)
            problem = acceptable(ev)
            if problem:
                trace.append(TuneStep(stack, ev, False, f"rejected: {problem}"))
                continue
            trace.append(TuneStep(stack, ev, False, "candidate"))
            if round_best is None or ev.paes > round_best[1].paes:
                round_best = (name, ev)

        if round_best is None:
            break
        name, ev = round_best
        if ev.paes <= best.paes * (1.0 + min_gain):
            # Nothing cleared the margin: record that the best of this round was close but not enough.
            for step in reversed(trace):
                if step.modules == [*best_modules, name]:
                    step.note = f"best of round, but gain {ev.paes / best.paes - 1:+.1%} <= required {min_gain:+.0%}"
                    break
            break
        best_modules.append(name)
        best = ev
        remaining.remove(name)
        for step in reversed(trace):
            if step.modules == best_modules:
                step.accepted = True
                step.note = f"accepted: PAES {ev.paes:.3f}"
                break

    report = None
    if constraints is not None:
        from worldoptbench.constraints import Measured, check_constraints

        report = check_constraints(
            constraints,
            Measured(physics=best.physics, speedup=best.speedup, vram_gb=best.vram_gb, latency_seconds=best.latency_seconds),
        )
    return TuneResult(best_modules=best_modules, best=best, baseline=baseline, trace=trace, report=report)


def evaluate_with_runner(
    model_factory: Callable[[], Any],
    run_kwargs: dict[str, Any] | None = None,
    *,
    repeats: int = 3,
    module_kwargs: dict[str, dict[str, Any]] | None = None,
) -> Evaluate:
    """Builds the real `evaluate`: a fresh model per run, the stack applied, the
    benchmark run `repeats` times, medians reported. Speedups are measured against
    the baseline's median latencies, which are taken on the first call (the empty
    stack), so `autotune` must evaluate the baseline first — it does.

    Process-global modules (TF32, distribution validation) are undone after
    every run via `OptimizationStack.restore()`, so one candidate can't leak into
    the next.
    """
    from worldoptbench.runner import latency_key, run_benchmark
    from worldoptbench.stack import OptimizationStack

    run_kwargs = dict(run_kwargs or {})
    baseline_latency: dict[str, float] = {}

    def evaluate(module_names: Sequence[str]) -> Evaluation:
        runs = []
        for _ in range(repeats):
            stack = OptimizationStack(model_factory(), list(module_names), module_kwargs=module_kwargs)
            try:
                results = run_benchmark(
                    stack.apply(),
                    baseline_latencies=baseline_latency or None,
                    optimization=stack.name,
                    **run_kwargs,
                )
            finally:
                stack.restore()
            runs.append(results)

        # median latency per rollout across repeats
        merged_latency = {
            latency_key(r.prompt_id, r.horizon): st.median(
                run[i].speed["latency_seconds"] for run in runs
            )
            for i, r in enumerate(runs[0])
        }
        if not module_names:
            baseline_latency.update(merged_latency)

        from worldoptbench.metrics.paes import compute_paes
        from worldoptbench.runner import _primary_physics_score

        paes_values, physics_values, speedup_values = [], [], []
        for r in runs[0]:
            physics = _primary_physics_score(r.physics)
            if physics is None:
                continue
            speedup = baseline_latency.get(latency_key(r.prompt_id, r.horizon), merged_latency[latency_key(r.prompt_id, r.horizon)]) / merged_latency[latency_key(r.prompt_id, r.horizon)]
            drift = (r.physics or {}).get("drift_rate") or 0.0
            paes_values.append(compute_paes(speedup, physics, drift, r.horizon))
            physics_values.append(physics)
            speedup_values.append(speedup)
        if not paes_values:
            raise RuntimeError("no physics scores were produced, so PAES can't be computed for autotuning")
        vram = [v for r in runs[0] for v in (r.speed.get("vram_reserved_gb"), r.speed.get("vram_peak_gb")) if v is not None]
        return Evaluation(
            paes=st.mean(paes_values),
            physics=st.mean(physics_values),
            latency_seconds=st.mean(merged_latency.values()),
            speedup=st.mean(speedup_values),
            vram_gb=max(vram) if vram else None,
        )

    return evaluate


def default_candidates(model: Any) -> list[str]:
    """Registered modules that can be applied to `model` here, in registry order."""
    from worldoptbench.optimizations.base import get_module_class

    return [
        name
        for name in available_modules()
        if get_module_class(name).autotune_candidate and get_module_class(name)().incompatibility(model) is None
    ]
