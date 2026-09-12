import numpy as np
import pytest

from worldoptbench.metrics.visual import VisualMetrics, compute_visual_metrics


def test_identical_frames_are_perfectly_temporally_consistent():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    metrics = compute_visual_metrics([frame, frame, frame])

    assert isinstance(metrics, VisualMetrics)
    assert metrics.temporal_consistency == 1.0
    assert metrics.psnr is None  # no reference given
    assert metrics.ssim is None
    assert metrics.fvd is None


def test_single_frame_is_trivially_consistent():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    metrics = compute_visual_metrics([frame])
    assert metrics.temporal_consistency == 1.0


def test_wildly_different_frames_lower_consistency():
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    metrics = compute_visual_metrics([black, white, black, white])
    assert metrics.temporal_consistency < 0.5


def test_reference_frames_enable_psnr_ssim():
    pytest.importorskip("torchmetrics")
    pytest.importorskip("torch")

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    metrics = compute_visual_metrics([frame, frame], reference_frames=[frame, frame])

    assert metrics.psnr is not None
    assert metrics.ssim is not None
