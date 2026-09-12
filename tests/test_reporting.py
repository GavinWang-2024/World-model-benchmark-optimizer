import json

from worldoptbench.reporting import compare, load_results, summarize


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
