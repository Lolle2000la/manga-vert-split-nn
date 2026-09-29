"""End-to-end tests for the thin CLI wrapper."""

import json
import os
import subprocess
import sys

import pytest
from test_page_break_detector import CHECKPOINT, MODEL_CONFIG, make_image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

requires_model = pytest.mark.skipif(
    not (os.path.exists(CHECKPOINT) and os.path.exists(MODEL_CONFIG)),
    reason="page-break model files are not present",
)


def run_cli(*extra):
    return subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "detect_breaks.py"),
            "--checkpoint",
            CHECKPOINT,
            "--config",
            MODEL_CONFIG,
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@requires_model
def test_cli_outputs_schema(tmp_path):
    image_path = tmp_path / "page.png"
    make_image(768, 400).save(image_path)

    proc = run_cli("--image", str(image_path))
    assert proc.returncode == 0, proc.stderr

    result = json.loads(proc.stdout)
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


@requires_model
def test_cli_missing_image_reports_error():
    proc = run_cli("--image", "/does/not/exist.png")
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert "error" in payload
