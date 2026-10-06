"""worldserve: the request queue, batching, cache, and the HTTP layer, with a fake model (no GPU,
no model download). Opens a localhost socket for the HTTP tests."""

import base64
import io
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface
from worldoptbench.serve import (
    ServerBusy,
    ServerStopped,
    WorldModelServer,
    decode_array,
    encode_array,
    encode_frames,
    make_http_server,
)


class FakeModel(WorldModelInterface):
    def __init__(self, delay=0.0, batching=False):
        self.calls = []
        self.batch_calls = []
        self.delay = delay
        self.gate = None  # an Event a test can use to hold generate() open
        if batching:
            self.generate_batch = self._generate_batch

    def generate(self, prompt=None, init_frame=None, init_video=None, actions=None, horizon=2.0, **kwargs):
        if self.gate is not None:
            self.gate.wait(5)
        if prompt == "boom":
            raise RuntimeError("model exploded")
        if prompt is None and init_video is None:
            raise ValueError("needs a prompt")
        time.sleep(self.delay)
        self.calls.append({"prompt": prompt, "horizon": horizon, **kwargs})
        value = (len(prompt or "") + int(kwargs.get("seed", 0))) % 250
        return Rollout(frames=[np.full((4, 6, 3), value + i, dtype=np.uint8) for i in range(3)], fps=8.0,
                       metadata={"prompt": prompt, "seed": kwargs.get("seed", 0)})

    def _generate_batch(self, requests, seed=0):
        self.batch_calls.append((len(requests), seed))
        return [Rollout(frames=[np.zeros((2, 2, 3), dtype=np.uint8)] * 2, fps=4.0, metadata={"i": i}) for i, _ in enumerate(requests)]

    def get_info(self):
        return ModelInfo(name="fake", architecture="diffusion", param_count=7)


def _server(model=None, **kwargs):
    return WorldModelServer(model or FakeModel(), **kwargs).start()


# ---- wire encoding ---------------------------------------------------------------------------------


def test_arrays_round_trip_through_the_wire_encoding():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(decode_array(encode_array(array)), array)
    assert decode_array([encode_array(array), encode_array(array)])[1].shape == (3, 4)
    assert decode_array("a prompt") == "a prompt" and decode_array([1, 2]) == [1, 2]  # not arrays: untouched


def test_frames_encode_as_npy_and_decode_back_exactly():
    frames = [np.full((4, 6, 3), i, dtype=np.uint8) for i in range(3)]
    out = encode_frames(frames, "npy")
    assert out["frame_format"] == "npy"
    stacked = np.load(io.BytesIO(base64.b64decode(out["frames_npy_b64"])))
    assert stacked.shape == (3, 4, 6, 3) and np.array_equal(stacked[2], frames[2])


def test_frames_encode_as_png_when_pillow_is_available():
    pytest.importorskip("PIL")
    from PIL import Image

    frames = [np.full((4, 6, 3), 10 * i, dtype=np.uint8) for i in range(2)]
    out = encode_frames(frames)
    assert out["frame_format"] == "png" and len(out["frames"]) == 2
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(out["frames"][1]))))
    assert np.array_equal(decoded, frames[1])


# ---- the queue and the worker ---------------------------------------------------------------------------


def test_generate_returns_frames_latency_and_the_active_stack():
    server = _server()
    try:
        response = server.generate({"prompt": "a ball", "horizon": 2, "seed": 3, "format": "npy"})
    finally:
        server.stop()
    assert response["num_frames"] == 3 and response["fps"] == 8.0 and response["frame_format"] == "npy"
    assert response["latency_seconds"] >= 0 and response["queue_seconds"] >= 0 and response["batch_size"] == 1
    assert response["metadata"] == {"prompt": "a ball", "seed": 3}
    assert response["stack"] == {"applied": [], "skipped": []}
    assert server.model.calls == [{"prompt": "a ball", "horizon": 2.0, "seed": 3}]  # horizon -> float, seed -> int


def test_params_are_passed_through_and_unknown_fields_are_rejected():
    server = _server()
    try:
        server.generate({"prompt": "p", "params": {"guidance_scale": 7.5}, "format": "npy"})
        assert server.model.calls[-1]["guidance_scale"] == 7.5
        with pytest.raises(ValueError, match="unknown request fields"):
            server.submit({"prompt": "p", "guidence": 1})
        with pytest.raises(TypeError, match="JSON object"):
            server.submit(["not", "a", "dict"])
    finally:
        server.stop()


def test_a_full_queue_raises_server_busy_instead_of_growing():
    model = FakeModel()
    model.gate = threading.Event()
    server = _server(model, max_queue=2)
    try:
        first = server.submit({"prompt": "one", "format": "npy"})  # taken by the worker, blocked on the gate
        time.sleep(0.1)
        server.submit({"prompt": "two", "format": "npy"})
        server.submit({"prompt": "three", "format": "npy"})
        with pytest.raises(ServerBusy):
            server.submit({"prompt": "four", "format": "npy"})
        model.gate.set()
        assert first.result(5)["num_frames"] == 3
    finally:
        model.gate.set()
        server.stop()


def test_stopping_fails_queued_requests_and_refuses_new_ones():
    model = FakeModel()
    model.gate = threading.Event()
    server = _server(model, max_queue=4)
    running = server.submit({"prompt": "running", "format": "npy"})
    time.sleep(0.1)
    queued = server.submit({"prompt": "queued", "format": "npy"})
    stopper = threading.Thread(target=server.stop)
    stopper.start()
    time.sleep(0.1)
    model.gate.set()
    stopper.join(10)
    running.result(5)
    with pytest.raises(ServerStopped):
        queued.result(5)
    with pytest.raises(ServerStopped):
        server.submit({"prompt": "late"})


def test_a_model_error_reaches_the_caller_and_the_worker_survives():
    server = _server()
    try:
        with pytest.raises(RuntimeError, match="model exploded"):
            server.generate({"prompt": "boom"})
        assert server.generate({"prompt": "fine", "format": "npy"})["num_frames"] == 3  # still serving
        profile = server.profile()
    finally:
        server.stop()
    assert profile["errors"] == 1 and profile["requests"] == 1


def test_profile_counts_requests_and_reports_latency_percentiles():
    server = _server(FakeModel(delay=0.01))
    try:
        for i in range(5):
            server.generate({"prompt": f"p{i}", "format": "npy"})
        profile = server.profile()
    finally:
        server.stop()
    assert profile["requests"] == 5 and profile["errors"] == 0 and profile["batches"] == 5
    assert 0.01 <= profile["latency_p50_seconds"] <= profile["latency_p95_seconds"]
    assert profile["queue_depth"] == 0


# ---- cache -----------------------------------------------------------------------------------------------


def test_the_cache_serves_repeats_without_calling_the_model_and_keys_on_the_whole_request():
    server = _server(cache_size=8)
    try:
        first = server.generate({"prompt": "a ball", "seed": 1, "format": "npy"})
        second = server.generate({"prompt": "a ball", "seed": 1, "format": "npy"})
        other_seed = server.generate({"prompt": "a ball", "seed": 2, "format": "npy"})
        profile = server.profile()
    finally:
        server.stop()
    assert first["cached"] is False and second["cached"] is True and other_seed["cached"] is False
    assert second["frames_npy_b64"] == first["frames_npy_b64"]
    assert len(server.model.calls) == 2 and profile["cache_hits"] == 1


def test_the_cache_is_bounded():
    server = _server(cache_size=2)
    try:
        for seed in range(4):
            server.generate({"prompt": "p", "seed": seed, "format": "npy"})
        assert server.generate({"prompt": "p", "seed": 0, "format": "npy"})["cached"] is False  # evicted
        assert server.generate({"prompt": "p", "seed": 3, "format": "npy"})["cached"] is True  # still there
    finally:
        server.stop()


# ---- batching --------------------------------------------------------------------------------------------


def _video_request(frames=2, steps=3, size=4, seed=0):
    return {
        "init_video": [encode_array(np.zeros((size, size, 3), dtype=np.uint8)) for _ in range(frames)],
        "actions": encode_array(np.zeros((steps, 2), dtype=np.float32)),
        "horizon": 1.0, "seed": seed, "format": "npy",
    }


def test_compatible_queued_requests_run_as_one_batch():
    model = FakeModel(batching=True)
    server = _server(model, max_batch=4, batch_wait=0.3)
    try:
        futures = [server.submit(_video_request()) for _ in range(4)]
        responses = [f.result(5) for f in futures]
    finally:
        server.stop()
    assert model.batch_calls == [(4, 0)]  # one batched call for all four
    assert [r["batch_size"] for r in responses] == [4, 4, 4, 4]
    assert [r["metadata"]["i"] for r in responses] == [0, 1, 2, 3]  # results scattered back in order


def test_requests_of_different_shapes_are_batched_separately():
    model = FakeModel(batching=True)
    server = _server(model, max_batch=8, batch_wait=0.3)
    try:
        futures = [server.submit(_video_request(size=4)), server.submit(_video_request(size=8)),
                   server.submit(_video_request(size=4))]
        sizes = sorted(f.result(5)["batch_size"] for f in futures)
    finally:
        server.stop()
    assert sizes == [1, 2, 2]  # the two 4x4 requests batched together, the 8x8 one alone
    assert model.batch_calls == [(2, 0)]  # a lone request goes through generate(), not generate_batch()


def test_without_max_batch_a_batching_model_still_gets_single_calls():
    model = FakeModel(batching=True)
    server = _server(model, max_batch=1)
    try:
        server.generate({**_video_request(), "prompt": "x"})
    finally:
        server.stop()
    assert model.batch_calls == [] and len(model.calls) == 1


def test_a_text_request_is_never_batched():
    model = FakeModel(batching=True)
    server = _server(model, max_batch=4, batch_wait=0.05)
    try:
        server.generate({"prompt": "text only", "format": "npy"})
    finally:
        server.stop()
    assert model.batch_calls == [] and len(model.calls) == 1


# ---- the stack ------------------------------------------------------------------------------------------


def test_recommended_uses_the_libraries_default_and_reports_it(monkeypatch):
    import worldoptbench.defaults as defaults

    monkeypatch.setattr(defaults, "recommended_config", lambda model: ([], {}))
    server = WorldModelServer(FakeModel(), recommended=True).start()
    try:
        assert server.info()["stack"] == {"applied": [], "skipped": []}
    finally:
        server.stop()


def test_constructor_validates_its_limits():
    with pytest.raises(ValueError, match="max_queue"):
        WorldModelServer(FakeModel(), max_queue=0)
    with pytest.raises(ValueError, match="cache_size"):
        WorldModelServer(FakeModel(), cache_size=-1)


# ---- HTTP --------------------------------------------------------------------------------------------------


@pytest.fixture()
def http_server():
    server = _server(FakeModel())
    httpd = make_http_server(server, "127.0.0.1", 0, timeout=10)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    server.stop()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, json.loads(response.read())


def _post(url, payload, raw=None):
    request = urllib.request.Request(url, data=raw if raw is not None else json.dumps(payload).encode(), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_health_info_and_profile(http_server):
    assert _get(http_server + "/health")[1]["status"] == "ok"
    info = _get(http_server + "/info")[1]
    assert info["model"] == "fake" and info["stack"]["applied"] == []
    assert _get(http_server + "/profile")[1]["requests"] == 0


def test_http_generate_returns_the_response(http_server):
    status, body = _post(http_server + "/generate", {"prompt": "a ball", "seed": 2, "format": "npy"})
    assert status == 200 and body["num_frames"] == 3 and body["metadata"]["seed"] == 2
    assert _get(http_server + "/profile")[1]["requests"] == 1


def test_http_maps_errors_to_status_codes(http_server):
    assert _post(http_server + "/generate", None, raw=b"{not json")[0] == 400
    assert _post(http_server + "/generate", {"prompt": "p", "typo": 1})[0] == 400  # unknown field
    status, body = _post(http_server + "/generate", {"horizon": 1})  # the fake model needs a prompt
    assert status == 400 and "needs a prompt" in body["error"]
    assert _post(http_server + "/generate", {"prompt": "boom"})[0] == 500
    assert _post(http_server + "/nowhere", {})[0] == 404
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(http_server + "/nowhere")
    assert error.value.code == 404


# ---- outline-style construction, constraints, serve() ------------------------------------------------------


def test_stack_is_an_alias_for_modules_and_both_at_once_is_an_error():
    server = WorldModelServer(FakeModel(), stack=[])
    assert server.stack_report() == {"applied": [], "skipped": []}
    with pytest.raises(ValueError, match="not both"):
        WorldModelServer(FakeModel(), ["a"], stack=["b"])


def test_the_worldserve_alias_package_exports_the_same_server():
    import worldserve

    assert worldserve.WorldModelServer is WorldModelServer and worldserve.ServerBusy is ServerBusy


def test_responses_carry_a_speedup_when_a_baseline_and_a_horizon_are_known():
    # baseline 10 s of compute per generated second; this fake model takes ~0 s, so the speedup is large and positive
    server = _server(FakeModel(delay=0.02), baseline_seconds_per_video_second=10.0)
    try:
        with_horizon = server.generate({"prompt": "p", "horizon": 2, "format": "npy"})
        without_horizon = _server(FakeModel(), baseline_seconds_per_video_second=10.0)
        try:
            no_horizon = without_horizon.generate({"prompt": "p", "format": "npy"})
        finally:
            without_horizon.stop()
    finally:
        server.stop()
    assert with_horizon["speedup"] > 1.0
    assert no_horizon["speedup"] is None or no_horizon["speedup"] > 0  # the fake's metadata has no horizon


def test_without_a_baseline_there_is_no_speedup():
    server = _server()
    try:
        assert server.generate({"prompt": "p", "horizon": 2, "format": "npy"})["speedup"] is None
        assert server.profile()["speedup_mean"] is None
    finally:
        server.stop()


def test_constraints_are_reported_and_only_the_speedup_target_is_checked_while_serving():
    from worldoptbench.constraints import Constraints

    server = _server(
        FakeModel(delay=0.01), baseline_seconds_per_video_second=100.0,
        constraints=Constraints(min_physics_score=0.9, target_speedup=2.0, max_vram_gb=8.0),
    )
    try:
        assert server.info()["constraints"] == {"min_physics_score": 0.9, "target_speedup": 2.0, "max_vram_gb": 8.0}
        assert server.profile()["constraints"]["target_speedup"]["met"] is None  # nothing served yet
        server.generate({"prompt": "p", "horizon": 2, "format": "npy"})
        status = server.profile()["constraints"]
    finally:
        server.stop()
    assert status["target_speedup"]["met"] is True  # 100 s per video-second baseline vs milliseconds
    assert status["min_physics_score"]["met"] is None and "cannot be verified" in status["min_physics_score"]["note"]
    assert status["max_vram_gb"]["met"] is None


def test_serve_without_blocking_returns_a_working_http_server():
    server = WorldModelServer(FakeModel())
    httpd = server.serve(host="127.0.0.1", port=0, block=False, timeout=10)
    try:
        port = httpd.server_address[1]
        assert _get(f"http://127.0.0.1:{port}/health")[1]["status"] == "ok"
        status, body = _post(f"http://127.0.0.1:{port}/generate", {"prompt": "a ball", "format": "npy"})
        assert status == 200 and body["num_frames"] == 3
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.stop()


def test_a_nonpositive_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline_seconds_per_video_second"):
        WorldModelServer(FakeModel(), baseline_seconds_per_video_second=0)


# ---- warmup ---------------------------------------------------------------------------------------------------


def test_warmup_requests_run_at_start_before_traffic_and_are_not_counted():
    model = FakeModel()
    server = WorldModelServer(model, warmup=[{"prompt": "warm one", "horizon": 2}, {"prompt": "warm two", "horizon": 5}])
    assert model.calls == []  # nothing runs until start()
    server.start()
    try:
        assert [c["prompt"] for c in model.calls] == ["warm one", "warm two"]
        assert server.profile()["requests"] == 0 and server.profile()["latency_p50_seconds"] is None
        server.generate({"prompt": "real", "format": "npy"})
        assert [c["prompt"] for c in model.calls][-1] == "real"
    finally:
        server.stop()


def test_a_failing_warmup_request_stops_start_with_its_error_and_a_bad_one_is_rejected_at_construction():
    server = WorldModelServer(FakeModel(), warmup=[{"prompt": "boom"}])
    with pytest.raises(RuntimeError, match="model exploded"):
        server.start()
    with pytest.raises(ValueError, match="unknown request fields"):
        WorldModelServer(FakeModel(), warmup=[{"prompt": "p", "typo": 1}])


def test_start_warms_up_only_once():
    model = FakeModel()
    server = WorldModelServer(model, warmup=[{"prompt": "w"}])
    server.start()
    server.start()
    try:
        assert len(model.calls) == 1
    finally:
        server.stop()
