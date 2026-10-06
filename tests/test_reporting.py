import json

import pytest

from worldoptbench.reporting import compare, drift_stats, load_results, summarize


def _fake_result(paes=0.8, latency=1.0, speedup=1.0):
    return {
        "model_name": "fake-model",
        "architecture": "diffusion",
        "prompt_id": "p1",
        "domain": "robotics",
        "horizon": 4,
        "speed": {"latency_seconds": latency, "fps": 4.0, "vram_peak_gb": None, "speedup": speedup},
        "visual": {"temporal_consistency": 0.9, "psnr": None, "ssim": None, "fvd": None},
        "physics": None,
        "paes": paes,
    }


def test_load_results_roundtrips_json(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps([_fake_result()]))

    loaded = load_results(path)
    assert loaded[0]["model_name"] == "fake-model"


def test_summarize_averages_across_results():
    results = [_fake_result(paes=0.8, latency=1.0), _fake_result(paes=1.0, latency=2.0)]
    summary = summarize(results)

    assert summary["model"] == "fake-model"
    assert summary["n_runs"] == 2
    assert summary["mean_paes"] == 0.9
    assert summary["mean_latency_seconds"] == 1.5


def test_summarize_reports_optimization_label_defaulting_to_baseline():
    labelled = {**_fake_result(), "optimization": "worldcache"}
    assert summarize([labelled])["optimization"] == "worldcache"
    assert summarize([_fake_result()])["optimization"] == "baseline"


def test_summarize_empty_results():
    summary = summarize([])
    assert summary["n_runs"] == 0


def test_compare_multiple_result_files(tmp_path):
    path_a = tmp_path / "baseline.json"
    path_b = tmp_path / "optimized.json"
    path_a.write_text(json.dumps([_fake_result(paes=0.8)]))
    path_b.write_text(json.dumps([_fake_result(paes=1.2)]))

    rows = compare([path_a, path_b])

    assert len(rows) == 2
    assert rows[0]["mean_paes"] == 0.8
    assert rows[1]["mean_paes"] == 1.2


def _with_drift(prompt_id, drift_rate):
    row = _fake_result()
    row["prompt_id"] = prompt_id
    row["physics"] = {"drift_rate": drift_rate}
    return row


def test_drift_stats_uses_one_rate_per_prompt_not_one_per_rollout():
    # Four horizons of the same prompt share one fitted drift rate; it must count once.
    rows = [_with_drift("a", -0.01)] * 4 + [_with_drift("b", -0.03)] * 4
    stats = drift_stats(rows)
    assert stats["n_scenarios"] == 2
    assert stats["mean_drift_rate"] == pytest.approx(-0.02)
    assert stats["drift_rate_sem"] is not None


def test_drift_is_not_called_detectable_when_scenarios_disagree_in_sign():
    # The real first-run pattern: -0.006, +0.001, +0.005 per second.
    rows = [_with_drift("a", -0.006), _with_drift("b", 0.001), _with_drift("c", 0.005)]
    assert drift_stats(rows)["drift_detectable"] is False


def test_drift_is_called_detectable_when_scenarios_agree():
    rows = [_with_drift(p, r) for p, r in zip("abcde", (-0.010, -0.011, -0.009, -0.010, -0.012))]
    stats = drift_stats(rows)
    assert stats["drift_detectable"] is True
    assert stats["mean_drift_rate"] == pytest.approx(-0.0104)


def test_drift_stats_needs_three_scenarios_and_tolerates_missing_physics():
    assert drift_stats([_fake_result()])["mean_drift_rate"] is None  # physics is None
    two = drift_stats([_with_drift("a", -0.01), _with_drift("b", -0.02)])
    assert two["drift_detectable"] is None  # too few scenarios to say
    assert "mean_drift_rate" in summarize([_with_drift("a", -0.01)])
