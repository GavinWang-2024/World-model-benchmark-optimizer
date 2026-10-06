import json

import numpy as np
import pytest

from worldoptbench.metrics.paes import compute_paes
from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.runner import latency_key, load_standard_set, run_benchmark


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


def test_run_benchmark_records_optimization_label():
    default = run_benchmark(FakeWorldModel())
    assert all(r.optimization == "baseline" for r in default)

    labelled = run_benchmark(FakeWorldModel(), optimization="worldcache")
    assert all(r.optimization == "worldcache" for r in labelled)


def test_run_benchmark_loads_baseline_from_results_json(tmp_path):
    baseline_file = tmp_path / "baseline.json"
    baseline = run_benchmark(FakeWorldModel(), output_path=baseline_file)

    # Rewrite every baseline latency to 1000s so any real run looks much faster,
    # as if an optimized run were compared against a very slow baseline.
    rows = json.loads(baseline_file.read_text())
    for row in rows:
        row["speed"]["latency_seconds"] = 1000.0
    baseline_file.write_text(json.dumps(rows))

    results = run_benchmark(FakeWorldModel(), baseline_path=baseline_file, optimization="fast")
    assert len(results) == len(baseline)
    assert all(r.speed["speedup"] > 1.0 for r in results)


def test_latency_key_is_stable_across_int_and_float_horizons():
    assert latency_key("p", 4) == latency_key("p", 4.0) == "p@4"


class _HorizonEncodingModel(FakeWorldModel):
    """Encodes the requested horizon into the frame's pixel values so a fake
    physics scorer (which only sees frames) can produce horizon-dependent scores.
    """

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        frame = np.full((4, 4, 3), int(horizon), dtype=np.uint8)
        return Rollout(frames=[frame, frame], fps=1.0)


def _patch_physics(monkeypatch, score_fn):
    from worldoptbench.metrics import physics

    monkeypatch.setattr(
        physics,
        "compute_physics_score",
        lambda frames, actions=None, reference_frames=None, baseline_frame=None: physics.PhysicsScore(pai_bench_score=score_fn(int(frames[0][0, 0, 0]))),
    )


def test_run_benchmark_fits_drift_per_prompt_and_penalizes_paes(monkeypatch):
    # Score falls from 0.98 at 4s to 0.40 at 120s.
    _patch_physics(monkeypatch, lambda horizon: 1.0 - horizon / 200)

    results = run_benchmark(_HorizonEncodingModel())

    for r in results:
        assert r.physics["drift_rate"] < 0
        assert r.physics["drift_onset"] == 60  # first horizon with score < 0.75
        expected = compute_paes(
            speedup=1.0,
            physics_score=r.physics["pai_bench_score"],
            drift_rate=r.physics["drift_rate"],
            horizon_t=r.horizon,
        )
        assert r.paes == pytest.approx(expected)
        assert r.paes < r.physics["pai_bench_score"]  # drift made it strictly worse


def test_run_benchmark_without_drift_matches_plain_score(monkeypatch):
    _patch_physics(monkeypatch, lambda horizon: 0.9)

    results = run_benchmark(_HorizonEncodingModel())

    for r in results:
        assert r.physics["drift_rate"] == pytest.approx(0.0, abs=1e-9)
        assert r.paes == pytest.approx(0.9)


def _stub_visual_metrics(monkeypatch):
    # With reference frames, the real compute_visual_metrics imports
    # torch/torchmetrics for PSNR/SSIM — not installed in a plain dev setup.
    from worldoptbench.metrics.visual import VisualMetrics

    monkeypatch.setattr(
        "worldoptbench.runner.compute_visual_metrics",
        lambda frames, reference_frames=None: VisualMetrics(temporal_consistency=1.0),
    )


class _KwargsRecordingModel(FakeWorldModel):
    def __init__(self):
        self.calls = []

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=4.0, **kwargs):
        self.calls.append({"horizon": horizon, "kwargs": kwargs})
        return Rollout(frames=[np.full((4, 4, 3), 10, dtype=np.uint8)] * 2, fps=1.0)


def test_run_benchmark_passes_scenario_kwargs_and_uses_its_reference_for_physics(monkeypatch):
    _stub_visual_metrics(monkeypatch)
    from worldoptbench.scenarios import Scenario

    seen = []

    def scenario_fn(entry, horizon):
        seen.append((entry["id"], horizon))
        return Scenario(
            generate_kwargs={"marker": f"{entry['id']}@{horizon}"},
            # Identical to the model output, so fidelity should be perfect.
            reference_frames=[np.full((4, 4, 3), 10, dtype=np.uint8)] * 2,
        )

    model = _KwargsRecordingModel()
    results = run_benchmark(model, scenario_fn=scenario_fn, warmup_runs=0)

    assert len(seen) == len(results) == len(model.calls)
    assert model.calls[0]["kwargs"]["marker"] == f"{seen[0][0]}@{seen[0][1]}"
    for r in results:
        assert r.physics["sim_fidelity_score"] == pytest.approx(1.0)
        assert r.paes == pytest.approx(1.0)  # speedup 1, fidelity 1, flat drift


def test_scenario_reference_takes_precedence_over_reference_by_prompt(monkeypatch):
    _stub_visual_metrics(monkeypatch)
    from worldoptbench.scenarios import Scenario

    def scenario_fn(entry, horizon):
        return Scenario(reference_frames=[np.full((4, 4, 3), 10, dtype=np.uint8)] * 2)

    # Wrong length on purpose: if this were used, fidelity scoring would raise.
    wrong_length = {
        p["id"]: [np.zeros((4, 4, 3), dtype=np.uint8)] * 5 for p in load_standard_set()["prompts"]
    }

    results = run_benchmark(
        _KwargsRecordingModel(), scenario_fn=scenario_fn, reference_frames_by_prompt=wrong_length
    )
    assert all(r.physics["sim_fidelity_score"] == pytest.approx(1.0) for r in results)


def test_warmup_runs_cover_every_horizon_untimed_and_are_not_reported():
    horizons = load_standard_set()["horizons_seconds"]

    model = _KwargsRecordingModel()
    results = run_benchmark(model, warmup_runs=2)
    assert len(model.calls) == len(results) + 2 * len(horizons)
    # Each warm-up pass visits every horizon (so per-shape setup is paid up front).
    assert [c["horizon"] for c in model.calls[: len(horizons)]] == list(horizons)

    model = _KwargsRecordingModel()
    results = run_benchmark(model, warmup_runs=0)
    assert len(model.calls) == len(results)


def test_run_benchmark_does_not_penalize_a_rising_physics_score(monkeypatch):
    # Score rises with horizon (as an under-trained model's cumulative-average skill
    # did on a real run): that is not drift, so PAES must equal the plain score.
    _patch_physics(monkeypatch, lambda horizon: 0.3 + horizon / 400)

    results = run_benchmark(_HorizonEncodingModel())

    for r in results:
        assert r.physics["drift_rate"] > 0
        assert r.paes == pytest.approx(r.physics["pai_bench_score"])
