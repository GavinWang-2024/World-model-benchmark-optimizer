"""Benchmark runner — orchestrates rollout generation + metric computation
for any WorldModelInterface implementation (outline §3.2, build_plan.md
Phase 2).

Physics scoring (Phase 3) and PAES (Phase 4) are wired in opportunistically:
if worldoptbench.metrics.physics still raises NotImplementedError because
PAI-Bench/WorldRoamBench aren't hooked up yet, the runner degrades to a
speed+visual-only result instead of failing the whole run. Once Phase 3
fills those stubs in, the exact same runner starts producing full profiles
with no changes needed here.

Drift is computed per prompt, not per rollout: a drift rate needs physics
scores at 2+ horizons, so scoring is done for every horizon of a prompt
first, then a drift curve is fit across them, then PAES is computed for each
rollout using that prompt's drift rate.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from worldoptbench.metrics.speed import measure_speed
from worldoptbench.metrics.visual import compute_visual_metrics
from worldoptbench.models.base import WorldModelInterface
from worldoptbench.scenarios import Scenario, ScenarioFn

_STANDARD_SET_PATH = Path(__file__).parent / "prompts" / "standard_set.json"

BASELINE_OPTIMIZATION = "baseline"


@dataclass
class RunResult:
    model_name: str
    architecture: str
    prompt_id: str
    domain: str
    horizon: float
    speed: dict[str, Any]
    visual: dict[str, Any]
    physics: dict[str, Any] | None = None
    paes: float | None = None
    optimization: str = BASELINE_OPTIMIZATION


def load_standard_set(path: Path = _STANDARD_SET_PATH) -> dict:
    return json.loads(Path(path).read_text())


def latency_key(prompt_id: str, horizon: float) -> str:
    """Key format shared by `baseline_latencies` and `load_baseline_latencies`.
    `:g` so 4 and 4.0 produce the same key (JSON round-trips can change which).
    """
    return f"{prompt_id}@{horizon:g}"


def load_baseline_latencies(path: Path) -> dict[str, float]:
    """Reads a results JSON written by a previous `run_benchmark` call into
    the `{"{prompt_id}@{horizon}": seconds}` dict `run_benchmark` uses for
    speedup.
    """
    return {
        latency_key(r["prompt_id"], r["horizon"]): r["speed"]["latency_seconds"]
        for r in json.loads(Path(path).read_text())
    }


def _score_physics(
    frames: list[Any],
    reference_frames: list[Any] | None = None,
    baseline_frame: Any | None = None,
) -> dict[str, Any] | None:
    # Imported at call time (not module level) so Phase 3 can swap the
    # implementation in without this module caring.
    from worldoptbench.metrics.physics import compute_physics_score

    try:
        return asdict(
            compute_physics_score(
                frames, reference_frames=reference_frames, baseline_frame=baseline_frame
            )
        )
    except NotImplementedError:
        return None  # Phase 3 physics wrappers not filled in yet — fine, degrade gracefully


def _primary_physics_score(physics: dict[str, Any] | None) -> float | None:
    """The 0-1 score PAES and drift fitting use: PAI-Bench when present,
    otherwise simulator fidelity (reference-backed models like Dreamer).
    """
    if not physics:
        return None
    for key in ("pai_bench_score", "sim_fidelity_score"):
        if physics.get(key) is not None:
            return physics[key]
    return None


def _apply_drift_and_paes(prompt_results: list[RunResult]) -> None:
    """Fits one drift curve across a prompt's horizons, writes drift_rate /
    drift_onset onto each rollout's physics dict, and computes PAES.

    With only one scored horizon there's no slope to fit, so drift_rate stays
    None and PAES is computed with no drift penalty.
    """
    from worldoptbench.metrics.paes import compute_paes
    from worldoptbench.metrics.physics import compute_drift_curve

    scores_by_horizon = {
        r.horizon: _primary_physics_score(r.physics)
        for r in prompt_results
        if _primary_physics_score(r.physics) is not None
    }
    curve = compute_drift_curve(scores_by_horizon) if len(scores_by_horizon) >= 2 else None

    for r in prompt_results:
        if r.horizon not in scores_by_horizon:
            continue
        if curve is not None:
            r.physics["drift_rate"] = curve.drift_rate
            r.physics["drift_onset"] = curve.drift_onset
        r.paes = compute_paes(
            speedup=r.speed["speedup"],
            physics_score=scores_by_horizon[r.horizon],
            drift_rate=r.physics.get("drift_rate") or 0.0,
            horizon_t=r.horizon,
        )


def run_benchmark(
    model: WorldModelInterface,
    standard_set_path: Path = _STANDARD_SET_PATH,
    output_path: Path | None = None,
    reference_frames_by_prompt: dict[str, list] | None = None,
    baseline_latencies: dict[str, float] | None = None,
    baseline_path: Path | None = None,
    optimization: str = BASELINE_OPTIMIZATION,
    scenario_fn: ScenarioFn | None = None,
    warmup_runs: int = 1,
    perceptual_scorer: Any = None,
) -> list[RunResult]:
    """Runs the standard prompt set against `model` at every horizon.

    Args:
        perceptual_scorer: optional `metrics.perceptual.PerceptualScorer` (or anything with its `.score(frames,
            reference_frames, prompt)`); when given, learned DINO/CLIP scores are added to each result's `visual`.
            Computed after all timing, so it cannot affect latency.
        model: any WorldModelInterface implementation.
        standard_set_path: defaults to the bundled prompts/standard_set.json.
        output_path: if given, results are also written here as JSON.
        reference_frames_by_prompt: optional {prompt_id: frames} ground-truth
            continuations, enabling PSNR/SSIM (see metrics/visual.py — these
            are skipped without a reference, which is the common case for
            pure Text2World prompts).
        baseline_latencies: optional {"{prompt_id}@{horizon}": seconds}
            (see `latency_key`) for computing `speedup`. Entries here
            override the same keys loaded from `baseline_path`.
        baseline_path: results JSON from a prior unoptimized run, loaded via
            `load_baseline_latencies`. If neither this nor `baseline_latencies`
            is given, speedup is 1.0 (this run *is* the baseline). If one is
            given but misses a (prompt, horizon) pair, that pair gets
            speedup 1.0 and a warning.
        optimization: label recorded on every result (e.g. "baseline",
            "worldcache", "worldcache+fp8") so result files stay
            distinguishable once you're comparing several.
        scenario_fn: optional (prompt entry, horizon) -> Scenario, for models
            that need more than a prompt (context frames, actions) and/or can
            supply ground truth — see worldoptbench.scenarios. Its
            `generate_kwargs` are passed to `model.generate()`, and its
            `reference_frames` take precedence over `reference_frames_by_prompt`.
            All scenarios are built up front, before any warm-up or timing, so
            building one (e.g. stepping a simulator) never counts toward
            latency or perturbs a neighboring timed call.
        warmup_runs: passes of untimed, discarded `generate()` calls — one per
            horizon, using the first prompt — before measuring. Without this the
            first timed rollout also pays for lazy weight loading and CUDA/cuDNN
            warm-up (on a real Dreamer run: 3.5 s against 0.15-0.5 s), and any
            per-shape setup such as CUDA-graph capture lands in the first
            rollout of each new horizon. Set 0 only if the model is already warm.

    Returns:
        One RunResult per (prompt, horizon) pair.
    """
    standard_set = load_standard_set(standard_set_path)
    horizons = standard_set["horizons_seconds"]
    info = model.get_info()

    baselines: dict[str, float] = {}
    if baseline_path is not None:
        baselines.update(load_baseline_latencies(baseline_path))
    if baseline_latencies:
        baselines.update(baseline_latencies)

    results: list[RunResult] = []

    # Build every scenario before any timing. Building one (e.g. creating a
    # simulator env with an OpenGL context) leaves the CPU/GPU busy, and on a
    # real run that doubled the *eager* baseline's time (62 -> 125 ms at 2 s)
    # when done right before a timed call, flattering any speedup measured
    # against it.
    scenarios: dict[tuple[str, float], Scenario] = {
        (entry["id"], horizon): scenario_fn(entry, horizon) if scenario_fn else Scenario()
        for entry in standard_set["prompts"]
        for horizon in horizons
    }

    warm_entry = standard_set["prompts"][0]
    for _ in range(warmup_runs):
        for warm_horizon in horizons:
            model.generate(
                prompt=warm_entry["prompt"],
                horizon=warm_horizon,
                **scenarios[(warm_entry["id"], warm_horizon)].generate_kwargs,
            )

    # Phase 1: time every rollout back to back, doing nothing else in between.
    # Computing metrics between timed calls is not free for the timing: PSNR/SSIM
    # and physics scoring are CPU-heavy, and on a real run they slowed the *next*
    # TensorRT-in-CUDA-graph call by ~5 ms (30.5 vs 25.8 ms at a 20 s horizon)
    # while barely touching a PyTorch call — so they quietly hid an optimization's
    # gain, unevenly. Frames are kept in memory until phase 2.
    timed: list[tuple[dict, float, Scenario, Any, Any]] = []  # (entry, horizon, scenario, rollout, speed)
    for entry in standard_set["prompts"]:
        for horizon in horizons:
            scenario = scenarios[(entry["id"], horizon)]
            rollout, speed = measure_speed(
                lambda entry=entry, horizon=horizon, scenario=scenario: model.generate(
                    prompt=entry["prompt"], horizon=horizon, **scenario.generate_kwargs
                )
            )

            baseline_latency = baselines.get(latency_key(entry["id"], horizon))
            if baselines and baseline_latency is None:
                warnings.warn(
                    f"No baseline latency for {latency_key(entry['id'], horizon)}; "
                    "speedup defaults to 1.0 for this rollout",
                    stacklevel=2,
                )
            speed.speedup = (
                baseline_latency / speed.latency_seconds
                if baseline_latency and speed.latency_seconds > 0
                else 1.0
            )
            timed.append((entry, horizon, scenario, rollout, speed))

    # Phase 2: metrics, now that nothing is being timed.
    for entry in standard_set["prompts"]:
        prompt_results: list[RunResult] = []
        for e, horizon, scenario, rollout, speed in timed:
            if e is not entry:
                continue
            reference = scenario.reference_frames
            if reference is None:
                reference = (reference_frames_by_prompt or {}).get(entry["id"])
            # the learned-score arguments are passed only when asked for, so callers (and test doubles) of the
            # older two-argument form keep working
            extra = {"prompt": entry.get("prompt"), "scorer": perceptual_scorer} if perceptual_scorer is not None else {}
            visual = compute_visual_metrics(rollout.frames, reference_frames=reference, **extra)

            prompt_results.append(
                RunResult(
                    model_name=info.name,
                    architecture=info.architecture,
                    prompt_id=entry["id"],
                    domain=entry.get("domain", "unknown"),
                    horizon=horizon,
                    speed=asdict(speed),
                    visual=asdict(visual),
                    physics=_score_physics(rollout.frames, reference, scenario.baseline_frame),
                    optimization=optimization,
                )
            )

        _apply_drift_and_paes(prompt_results)
        results.extend(prompt_results)

    if output_path:
        Path(output_path).write_text(json.dumps([asdict(r) for r in results], indent=2))

    return results
