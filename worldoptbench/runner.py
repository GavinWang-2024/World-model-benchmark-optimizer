"""Benchmark runner — orchestrates rollout generation + metric computation
for any WorldModelInterface implementation (outline §3.2, build_plan.md
Phase 2).

Physics scoring (Phase 3) and PAES (Phase 4) are wired in opportunistically:
if worldoptbench.metrics.physics still raises NotImplementedError because
PAI-Bench/WorldRoamBench aren't hooked up yet, the runner degrades to a
speed+visual-only result instead of failing the whole run. Once Phase 3
fills those stubs in, the exact same runner starts producing full profiles
with no changes needed here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from worldoptbench.metrics.speed import measure_speed
from worldoptbench.metrics.visual import compute_visual_metrics
from worldoptbench.models.base import WorldModelInterface

_STANDARD_SET_PATH = Path(__file__).parent / "prompts" / "standard_set.json"


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


def load_standard_set(path: Path = _STANDARD_SET_PATH) -> dict:
    return json.loads(Path(path).read_text())


def run_benchmark(
    model: WorldModelInterface,
    standard_set_path: Path = _STANDARD_SET_PATH,
    output_path: Path | None = None,
    reference_frames_by_prompt: dict[str, list] | None = None,
    baseline_latencies: dict[str, float] | None = None,
) -> list[RunResult]:
    """Runs the standard prompt set against `model` at every horizon.

    Args:
        model: any WorldModelInterface implementation.
        standard_set_path: defaults to the bundled prompts/standard_set.json.
        output_path: if given, results are also written here as JSON.
        reference_frames_by_prompt: optional {prompt_id: frames} ground-truth
            continuations, enabling PSNR/SSIM (see metrics/visual.py — these
            are skipped without a reference, which is the common case for
            pure Text2World prompts).
        baseline_latencies: optional {"{prompt_id}@{horizon}": seconds} from
            a prior unoptimized run, used to compute `speedup` for PAES. If
            not given, speedup defaults to 1.0 (i.e. this run *is* the
            baseline) — save its output and pass it back in here on the next
            (optimized) run to get real speedup numbers.

    Returns:
        One RunResult per (prompt, horizon) pair.
    """
    standard_set = load_standard_set(standard_set_path)
    horizons = standard_set["horizons_seconds"]
    info = model.get_info()
    results: list[RunResult] = []

    for entry in standard_set["prompts"]:
        for horizon in horizons:
            rollout, speed = measure_speed(
                lambda entry=entry, horizon=horizon: model.generate(
                    prompt=entry["prompt"], horizon=horizon
                )
            )

            key = f"{entry['id']}@{horizon}"
            baseline_latency = (baseline_latencies or {}).get(key)
            speed.speedup = (
                baseline_latency / speed.latency_seconds
                if baseline_latency and speed.latency_seconds > 0
                else 1.0
            )

            reference = (reference_frames_by_prompt or {}).get(entry["id"])
            visual = compute_visual_metrics(rollout.frames, reference_frames=reference)

            physics = None
            paes = None
            try:
                from worldoptbench.metrics.paes import compute_paes
                from worldoptbench.metrics.physics import compute_physics_score

                physics = compute_physics_score(rollout.frames)
                if physics.pai_bench_score is not None:
                    paes = compute_paes(
                        speedup=speed.speedup,
                        physics_score=physics.pai_bench_score,
                        drift_rate=physics.drift_rate or 0.0,
                        horizon_t=horizon,
                    )
            except NotImplementedError:
                pass  # Phase 3 physics wrappers not filled in yet — fine, degrade gracefully

            results.append(
                RunResult(
                    model_name=info.name,
                    architecture=info.architecture,
                    prompt_id=entry["id"],
                    domain=entry.get("domain", "unknown"),
                    horizon=horizon,
                    speed=asdict(speed),
                    visual=asdict(visual),
                    physics=asdict(physics) if physics else None,
                    paes=paes,
                )
            )

    if output_path:
        Path(output_path).write_text(json.dumps([asdict(r) for r in results], indent=2))

    return results
