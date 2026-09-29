"""Tests for the reusable page-break detection logic."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from page_break_detector import (
    DEFAULT_PARAMETERS,
    PageBreakDetectionError,
    PageBreakDetector,
    predict_sliding_window,
    resolve_parameters,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(
    REPO_ROOT, "models", "BCE Only (v8)", "final_deployment", "best_model.pth"
)
MODEL_CONFIG = os.path.join(
    REPO_ROOT, "models", "BCE Only (v8)", "final_deployment", "model_config.json"
)


class SpikeModel(nn.Module):
    """Deterministic stub: a flat baseline with spikes at fixed y positions."""

    def __init__(self, spikes, value=10.0, baseline=0.0):
        super().__init__()
        self.spikes = list(spikes)
        self.value = value
        self.baseline = baseline

    def forward(self, x):
        b, _c, h, _w = x.shape
        out = torch.full((b, h), self.baseline, device=x.device)
        for pos in self.spikes:
            if 0 <= pos < h:
                out[:, pos] = self.value
        return out


def make_image(width=768, height=512):
    arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# --------------------------------------------------------------------------- #
# Parameter resolution
# --------------------------------------------------------------------------- #


def test_resolve_parameters_defaults():
    params = resolve_parameters(None, None)
    assert params == DEFAULT_PARAMETERS


def test_resolve_parameters_config_then_overrides():
    config = {"peak_height": 0.8, "peak_distance": 20, "smoothing_sigma": 2.0}
    params = resolve_parameters(config, {"peak_height": 0.9, "edge_margin": 10})
    assert params["peak_height"] == 0.9  # override wins
    assert params["peak_distance"] == 20  # from config
    assert params["smoothing_sigma"] == 2.0  # from config
    assert params["peak_prominence"] == DEFAULT_PARAMETERS["peak_prominence"]
    assert params["edge_margin"] == 10  # override


def test_resolve_parameters_none_override_falls_through():
    config = {"peak_height": 0.7}
    params = resolve_parameters(config, {"peak_height": None})
    assert params["peak_height"] == 0.7


# --------------------------------------------------------------------------- #
# Sliding window
# --------------------------------------------------------------------------- #


def test_predict_sliding_window_single_chunk_matches_model():
    model = SpikeModel([100])
    x = torch.zeros(1, 3, 512, 768)
    logits = predict_sliding_window(model, x, chunk_size=2048, overlap=0)
    assert logits.shape == (1, 512)
    assert logits[0, 100].item() == pytest.approx(10.0)
    assert logits[0, 0].item() == pytest.approx(0.0)


def test_predict_sliding_window_aborts_between_chunks():
    model = SpikeModel([100])
    x = torch.zeros(1, 3, 3000, 768)
    calls = {"n": 0}

    def should_abort():
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(PageBreakDetectionError):
        predict_sliding_window(
            model, x, chunk_size=1000, overlap=0, should_abort=should_abort
        )


# --------------------------------------------------------------------------- #
# Detector end-to-end with a stub model
# --------------------------------------------------------------------------- #


def test_detector_reports_peaks_at_expected_positions():
    detector = PageBreakDetector(
        model=SpikeModel([100, 300]),
        parameters={
            "peak_height": 0.5,
            "peak_distance": 10,
            "smoothing_sigma": 0.0,
            "peak_prominence": 0.1,
            "edge_margin": 0,
        },
        device=torch.device("cpu"),
        target_width=768,
        chunk_size=2048,
        overlap=0,
    )

    result = detector.detect_image(make_image(768, 512), "page.png")

    assert result["image"] == "page.png"
    assert result["original_width"] == 768
    assert result["original_height"] == 512
    assert result["resized_height"] == 512
    assert result["scale_factor"] == pytest.approx(1.0)
    assert result["count"] == 2
    assert [s["y_resized"] for s in result["splits"]] == [100, 300]
    assert [s["y_original"] for s in result["splits"]] == [100, 300]
    assert all(0.99 < s["confidence"] <= 1.0 for s in result["splits"])


def test_detector_scales_coordinates_back_to_original():
    detector = PageBreakDetector(
        model=SpikeModel([100]),
        parameters={
            "peak_height": 0.5,
            "peak_distance": 10,
            "smoothing_sigma": 0.0,
            "peak_prominence": 0.1,
            "edge_margin": 0,
        },
        device=torch.device("cpu"),
        target_width=768,
        chunk_size=2048,
        overlap=0,
    )
    # Half the target width -> scale 2.0, so resized y=100 maps back to y=50.
    result = detector.detect_image(make_image(384, 400), "small.png")
    assert result["scale_factor"] == pytest.approx(2.0)
    assert result["splits"][0]["y_resized"] == 100
    assert result["splits"][0]["y_original"] == 50


def test_detector_detect_bytes_matches_detect_path(tmp_path):
    detector = PageBreakDetector(
        model=SpikeModel([64]),
        parameters={
            "peak_height": 0.5,
            "peak_distance": 10,
            "smoothing_sigma": 0.0,
            "peak_prominence": 0.1,
            "edge_margin": 0,
        },
        device=torch.device("cpu"),
        target_width=768,
        chunk_size=2048,
        overlap=0,
    )
    image = make_image(768, 256)
    path = tmp_path / "p.png"
    image.save(path)

    from_path = detector.detect_path(str(path))
    from_bytes = detector.detect_bytes(path.read_bytes(), "p.png")
    assert from_path["splits"] == from_bytes["splits"]


def test_detector_missing_file_raises():
    detector = PageBreakDetector(
        model=SpikeModel([1]),
        device=torch.device("cpu"),
    )
    with pytest.raises(PageBreakDetectionError):
        detector.detect_path("/does/not/exist.png")


# --------------------------------------------------------------------------- #
# Real checkpoint (skipped when the models are not present)
# --------------------------------------------------------------------------- #

real_model = pytest.mark.skipif(
    not (os.path.exists(CHECKPOINT) and os.path.exists(MODEL_CONFIG)),
    reason="page-break model files are not present",
)


@real_model
def test_real_model_output_matches_schema():
    detector = PageBreakDetector.from_checkpoint(
        checkpoint_path=CHECKPOINT,
        config_path=MODEL_CONFIG,
        device="cpu",
        target_width=768,
    )
    result = detector.detect_image(make_image(768, 400), "real.png")

    required = {
        "image",
        "original_height",
        "original_width",
        "resized_height",
        "scale_factor",
        "parameters",
        "splits",
        "count",
    }
    assert required <= set(result)
    assert result["count"] == len(result["splits"])
    assert set(result["parameters"]) == {
        "peak_height",
        "peak_distance",
        "smoothing_sigma",
        "peak_prominence",
        "edge_margin",
    }
    # The result must round-trip through JSON for the NDJSON transport.
    json.dumps(result)
