import json

import numpy as np

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.runner import load_standard_set, run_benchmark


class FakeWorldModel(WorldModelInterface):
    """Returns a couple of tiny numpy frames — no torch needed — so runner
    orchestration can be tested without any real model or GPU.
    """

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        return Rollout(frames=[frame, frame, frame], fps=1.0)

    def get_info(self):
        return ModelInfo(name="fake-model", architecture="diffusion", param_count=0)


def test_load_standard_set_has_expected_shape():
    standard_set = load_standard_set()
    assert "horizons_seconds" in standard_set
    assert "prompts" in standard_set
    assert len(standard_set["prompts"]) > 0
    assert all("id" in p and "prompt" in p for p in standard_set["prompts"])


def test_run_benchmark_produces_one_result_per_prompt_per_horizon():
    standard_set = load_standard_set()
    n_prompts = len(standard_set["prompts"])
    n_horizons = len(standard_set["horizons_seconds"])

    results = run_benchmark(FakeWorldModel())

    assert len(results) == n_prompts * n_horizons
    assert all(r.model_name == "fake-model" for r in results)
    assert all(r.architecture == "diffusion" for r in results)


def test_run_benchmark_degrades_gracefully_without_physics_phase3():
    # Phase 3 (PAI-Bench/WorldRoamBench) isn't wired up yet, so physics/paes
    # should be None rather than the run failing outright.
    results = run_benchmark(FakeWorldModel())
    assert all(r.physics is None for r in results)
    assert all(r.paes is None for r in results)
    assert all(r.speed["latency_seconds"] >= 0 for r in results)
    assert all(0.0 <= r.visual["temporal_consistency"] <= 1.0 for r in results)


def test_run_benchmark_writes_output_json(tmp_path):
    output_path = tmp_path / "results.json"
    run_benchmark(FakeWorldModel(), output_path=output_path)

    written = json.loads(output_path.read_text())
    assert isinstance(written, list)
    assert len(written) > 0
    assert "prompt_id" in written[0]


def test_run_benchmark_uses_baseline_latencies_for_speedup():
    standard_set = load_standard_set()
    first_id = standard_set["prompts"][0]["id"]
    first_horizon = standard_set["horizons_seconds"][0]

    # Pretend the baseline took 10x longer than whatever this (instant, fake)
    # model takes, at just that one (prompt, horizon) pair.
    results = run_benchmark(FakeWorldModel())
    actual_latency = next(
        r.speed["latency_seconds"]
        for r in results
        if r.prompt_id == first_id and r.horizon == first_horizon
    )
    baseline_latencies = {f"{first_id}@{first_horizon}": actual_latency * 10}

    results_with_baseline = run_benchmark(FakeWorldModel(), baseline_latencies=baseline_latencies)
    target = next(
        r for r in results_with_baseline if r.prompt_id == first_id and r.horizon == first_horizon
    )
    assert target.speed["speedup"] > 1.0
