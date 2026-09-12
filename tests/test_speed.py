import time

from worldoptbench.metrics.speed import SpeedMetrics, measure_speed


def test_measure_speed_times_the_call():
    def slow_fn():
        time.sleep(0.01)
        return "result"

    result, speed = measure_speed(slow_fn)

    assert result == "result"
    assert isinstance(speed, SpeedMetrics)
    assert speed.latency_seconds >= 0.01
    assert speed.speedup == 1.0  # default, filled in by the caller


def test_measure_speed_computes_fps_from_frames():
    class FakeRollout:
        def __init__(self):
            self.frames = ["f1", "f2", "f3", "f4"]

    _result, speed = measure_speed(lambda: FakeRollout())

    assert speed.fps is not None
    assert speed.fps > 0


def test_measure_speed_fps_none_without_frames():
    _result, speed = measure_speed(lambda: "no frames attribute here")

    assert speed.fps is None


def test_vram_peak_is_none_or_float():
    # On a machine with no CUDA (e.g. this dev laptop), should degrade to
    # None rather than raise.
    _, speed = measure_speed(lambda: None)
    assert speed.vram_peak_gb is None or isinstance(speed.vram_peak_gb, float)
