"""Subprocess smoke test for detect_server.py with the real model."""

import json
import os
import queue
import subprocess
import sys
import threading

import pytest
from test_page_break_detector import CHECKPOINT, MODEL_CONFIG, make_image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

requires_model = pytest.mark.skipif(
    not (os.path.exists(CHECKPOINT) and os.path.exists(MODEL_CONFIG)),
    reason="page-break model files are not present",
)


class ServerClient:
    def __init__(self):
        self.proc = subprocess.Popen(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "detect_server.py"),
                "--checkpoint",
                CHECKPOINT,
                "--config",
                MODEL_CONFIG,
                "--device",
                "cpu",
            ],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in iter(self.proc.stdout.readline, ""):
            if line:
                self._q.put(line)

    def read(self, timeout=60):
        try:
            return json.loads(self._q.get(timeout=timeout))
        except queue.Empty:
            return None

    def send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def read_until(self, event_type, timeout=60):
        while True:
            event = self.read(timeout)
            if event is None:
                return None
            if event.get("type") == event_type:
                return event

    def shutdown(self):
        self.send({"type": "shutdown"})
        self.read_until("exited")
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@requires_model
def test_server_process_detects_and_exits(tmp_path):
    image_path = tmp_path / "page.png"
    make_image(768, 400).save(image_path)

    client = ServerClient()
    ready = client.read()
    assert ready["type"] == "ready", ready

    client.send({"type": "detect", "id": "p1", "path": str(image_path)})
    result = client.read_until("result")
    assert result is not None, "server did not return a result"
    assert result["id"] == "p1"
    assert result["result"]["count"] == len(result["result"]["splits"])

    client.send({"type": "ping"})
    assert client.read_until("pong") is not None

    client.shutdown()
