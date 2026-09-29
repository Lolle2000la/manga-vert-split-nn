"""Tests for the long-running NDJSON detection server."""

import base64
import io
import json

import torch
from test_page_break_detector import SpikeModel, make_image

from detect_server import DetectionServer
from page_break_detector import PageBreakDetector


def make_detector():
    return PageBreakDetector(
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


def run_server(lines, detector=None):
    stdin = io.StringIO("".join(json.dumps(line) + "\n" for line in lines))
    stdout = io.StringIO()
    server = DetectionServer(detector or make_detector(), stdin=stdin, stdout=stdout)
    server.run()
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def events_of_type(events, event_type):
    return [e for e in events if e.get("type") == event_type]


def test_ready_result_and_exit(tmp_path):
    image_path = tmp_path / "page.png"
    make_image(768, 256).save(image_path)

    events = run_server(
        [
            {"type": "detect", "id": "p1", "path": str(image_path)},
            {"type": "shutdown"},
        ]
    )

    assert events[0]["type"] == "ready"
    assert events[0]["device"] == "cpu"
    results = events_of_type(events, "result")
    assert len(results) == 1
    assert results[0]["id"] == "p1"
    assert results[0]["result"]["count"] == 1
    assert results[0]["result"]["splits"][0]["y_resized"] == 64
    assert events[-1]["type"] == "exited"


def test_detect_from_base64_data():
    image_bytes = io.BytesIO()
    make_image(768, 256).save(image_bytes, "PNG")
    encoded = base64.b64encode(image_bytes.getvalue()).decode("ascii")

    events = run_server(
        [
            {"type": "detect", "id": "b1", "data": encoded, "name": "b.png"},
            {"type": "shutdown"},
        ]
    )

    results = events_of_type(events, "result")
    assert len(results) == 1
    assert results[0]["result"]["image"] == "b.png"
    assert results[0]["result"]["count"] == 1


def test_ping_is_answered():
    events = run_server([{"type": "ping"}, {"type": "shutdown"}])
    assert len(events_of_type(events, "pong")) == 1


def test_release_cache_when_idle():
    events = run_server([{"type": "release_cache"}, {"type": "shutdown"}])
    released = events_of_type(events, "cache_released")
    assert len(released) == 1
    assert released[0]["status"] == "ok"


def test_cancel_before_detection_starts(tmp_path):
    image_path = tmp_path / "page.png"
    make_image(768, 256).save(image_path)

    events = run_server(
        [
            {"type": "detect", "id": "c1", "path": str(image_path)},
            {"type": "cancel", "id": "c1"},
            {"type": "shutdown"},
        ]
    )

    # The reader thread observes the cancel before the job loop starts the detect,
    # so the job is dropped rather than run.
    assert len(events_of_type(events, "cancelled")) >= 1
    assert len(events_of_type(events, "result")) == 0


def test_unknown_request_reports_error():
    events = run_server([{"type": "nonsense"}, {"type": "shutdown"}])
    errors = events_of_type(events, "error")
    assert any("unknown request type" in e["message"] for e in errors)


def test_detect_requires_id():
    events = run_server([{"type": "detect", "path": "/x.png"}, {"type": "shutdown"}])
    errors = events_of_type(events, "error")
    assert any("requires an id" in e["message"] for e in errors)


def test_detect_requires_path_or_data():
    events = run_server([{"type": "detect", "id": "n1"}, {"type": "shutdown"}])
    errors = events_of_type(events, "error")
    assert any("'path' or 'data'" in e["message"] for e in errors)


def test_bad_image_reports_error_not_crash():
    events = run_server(
        [
            {"type": "detect", "id": "e1", "path": "/does/not/exist.png"},
            {
                "type": "detect",
                "id": "e2",
                "data": base64.b64encode(b"not an image").decode("ascii"),
            },
            {"type": "shutdown"},
        ]
    )
    errors = events_of_type(events, "error")
    assert {e["id"] for e in errors} == {"e1", "e2"}
    assert events[-1]["type"] == "exited"
