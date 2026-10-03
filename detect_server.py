#!/usr/bin/env python3
"""
Long-running NDJSON page-break detection server.

The CLI (``detect_breaks.py``) reloads the model for every image, which
dominates the cost when detecting a whole chapter. This server keeps the model
resident and speaks newline-delimited JSON (NDJSON) over stdin/stdout:

* **stdin**  — one request per line (JSON object).
* **stdout** — one event per line (JSON object), always UTF-8 and flushed.
* **stderr** — human-readable logs only.

It mirrors ``MangaJaNaiConverterGui``'s ``worker.py``: a single client (the
process that spawned it), no network port, and a ``release_cache`` command so a
co-tenant GPU process (the upscaler) can use the VRAM while the detector is idle.

Requests
--------
``{"type": "detect", "id": "p1", "path": "/tmp/p1.png"}``
``{"type": "detect", "id": "p1", "data": "<base64>", "name": "p1.png"}``
``{"type": "cancel", "id": "p1"}``
``{"type": "release_cache"}``
``{"type": "ping"}``
``{"type": "shutdown"}``

Events
------
``ready``, ``result``, ``error``, ``cancelled``, ``cache_released``, ``pong``,
``exited``. Every event is a single-line JSON object.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import sys
import threading
import time
from collections.abc import Iterable
from typing import Any, TextIO

from page_break_detector import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TARGET_WIDTH,
    PageBreakDetectionError,
    PageBreakDetector,
    release_accelerator_cache,
)


class DetectionServer:
    """NDJSON server around a :class:`PageBreakDetector`.

    ``stdin``/``stdout`` are injectable so the protocol can be unit-tested
    without spawning a process. Inference is sequential (a single GPU model),
    while control messages (``cancel``/``ping``/``release_cache``) are handled by
    the reader thread so they stay responsive during a long detect.
    """

    def __init__(
        self,
        detector: PageBreakDetector,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        idle_cache_release: float = 0.0,
    ) -> None:
        self.detector = detector
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout
        self.idle_cache_release = max(0.0, float(idle_cache_release))

        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._jobs: queue.Queue[dict[str, Any]] = queue.Queue()
        self._cancelled: set[str] = set()
        self._current_id: str | None = None
        self._shutdown = False
        self._idle_since: float | None = None

    # -- output ------------------------------------------------------------- #

    def emit(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False)
        with self._write_lock:
            self._stdout.write(line + "\n")
            self._stdout.flush()

    # -- request handling --------------------------------------------------- #

    def _cancel(self, job_id: str | None) -> None:
        if job_id is None:
            self.emit({"type": "error", "id": None, "message": "cancel requires an id"})
            return
        with self._state_lock:
            # A cancel for a job that never runs (a stale or duplicated id) would otherwise
            # accumulate for the server's lifetime; bound the set. A legitimate pending cancel is
            # only dropped once thousands are outstanding, which cannot happen for a single client.
            if len(self._cancelled) >= 1024:
                self._cancelled.clear()
            self._cancelled.add(job_id)
        self.emit({"type": "cancelled", "id": job_id})

    def _release_cache(self) -> None:
        with self._state_lock:
            busy = self._current_id is not None
        if busy:
            # Freeing cached blocks mid-inference would only slow the running job.
            self.emit({"type": "cache_released", "status": "busy"})
            return
        release_accelerator_cache(self.detector.device)
        self.emit({"type": "cache_released", "status": "ok"})

    def handle_control(self, msg: dict[str, Any]) -> bool:
        """Handle a non-detect request. Returns False when the server must stop."""
        mtype = msg.get("type")
        if mtype == "cancel":
            self._cancel(msg.get("id"))
        elif mtype == "release_cache":
            self._release_cache()
        elif mtype == "ping":
            self.emit({"type": "pong"})
        elif mtype == "shutdown":
            self._shutdown = True
            self._jobs.put({"type": "__stop__"})
            return False
        else:
            self.emit(
                {
                    "type": "error",
                    "id": msg.get("id"),
                    "message": f"unknown request type: {mtype}",
                }
            )
        return True

    def submit(self, msg: dict[str, Any]) -> None:
        self._jobs.put(msg)

    def _is_cancelled(self, job_id: str | None) -> bool:
        with self._state_lock:
            return job_id is not None and job_id in self._cancelled

    def handle_detect(self, msg: dict[str, Any]) -> None:
        job_id = msg.get("id")
        if job_id is None:
            self.emit({"type": "error", "id": None, "message": "detect requires an id"})
            return

        with self._state_lock:
            cancelled = job_id in self._cancelled
            self._cancelled.discard(job_id)
            if not cancelled:
                self._current_id = job_id
                self._idle_since = None

        if cancelled:
            self.emit({"type": "cancelled", "id": job_id})
            return

        should_abort = lambda: self._is_cancelled(job_id)
        try:
            if msg.get("data") is not None:
                data = base64.b64decode(msg["data"])
                result = self.detector.detect_bytes(
                    data, msg.get("name", ""), should_abort
                )
            elif msg.get("path") is not None:
                result = self.detector.detect_path(msg["path"], should_abort)
            else:
                self.emit(
                    {
                        "type": "error",
                        "id": job_id,
                        "message": "detect requires either 'path' or 'data'",
                    }
                )
                return

            self.emit({"type": "result", "id": job_id, "result": result})
        except PageBreakDetectionError as exc:
            if str(exc) == "cancelled" or self._is_cancelled(job_id):
                self.emit({"type": "cancelled", "id": job_id})
            else:
                self.emit({"type": "error", "id": job_id, "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - never kill the server over one image
            self.emit(
                {
                    "type": "error",
                    "id": job_id,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            with self._state_lock:
                if self._current_id == job_id:
                    self._current_id = None
                    self._idle_since = time.monotonic()

    def _stdin_loop(self) -> None:
        for raw_line in self._stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                self.emit(
                    {"type": "error", "id": None, "message": f"invalid JSON: {exc}"}
                )
                continue
            if not isinstance(msg, dict):
                self.emit(
                    {
                        "type": "error",
                        "id": None,
                        "message": "request must be an object",
                    }
                )
                continue

            mtype = msg.get("type")
            if mtype == "detect":
                self.submit(msg)
            elif not self.handle_control(msg):
                break

        self._shutdown = True
        self._jobs.put({"type": "__stop__"})

    def _maybe_release_idle_cache(self) -> None:
        if self.idle_cache_release <= 0:
            return
        with self._state_lock:
            idle_since = self._idle_since
            busy = self._current_id is not None
        if busy or idle_since is None:
            return
        if time.monotonic() - idle_since >= self.idle_cache_release:
            release_accelerator_cache(self.detector.device)
            with self._state_lock:
                self._idle_since = None

    def run(self) -> None:
        self.emit(
            {
                "type": "ready",
                "device": str(self.detector.device),
                "target_width": self.detector.target_width,
            }
        )
        reader = threading.Thread(target=self._stdin_loop, daemon=True)
        reader.start()

        try:
            while True:
                try:
                    job = self._jobs.get(timeout=0.5)
                except queue.Empty:
                    self._maybe_release_idle_cache()
                    if self._shutdown:
                        break
                    continue

                if job.get("type") == "__stop__":
                    break
                self.handle_detect(job)
        finally:
            reader.join(timeout=5)
            self.emit({"type": "exited"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python detect_server.py",
        description="Long-running NDJSON page-break detection server.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config (.json)"
    )
    parser.add_argument(
        "--peak-height",
        type=float,
        default=None,
        help="Minimum height for a peak (0.0-1.0)",
    )
    parser.add_argument(
        "--peak-distance",
        type=int,
        default=None,
        help="Minimum distance between peaks (pixels)",
    )
    parser.add_argument(
        "--smoothing-sigma", type=float, default=None, help="Gaussian smoothing sigma"
    )
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=None,
        help="Minimum prominence of peaks",
    )
    parser.add_argument(
        "--edge-margin",
        type=int,
        default=None,
        help="Pixels from top/bottom to ignore splits (in resized coordinates)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, e.g. 'cuda:1' or 'cpu'.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=DEFAULT_TARGET_WIDTH,
        help="Width the image is resized to before inference.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Sliding-window chunk height in pixels.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Sliding-window overlap in pixels.",
    )
    parser.add_argument(
        "--idle-cache-release",
        type=float,
        default=0.0,
        help="Seconds of idleness after which cached VRAM is returned to the driver (0 disables).",
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help=(
            "Exit when this parent process dies, so an abruptly killed driver does not "
            "leave a warm model resident on the GPU."
        ),
    )
    return parser


def _die_with_parent(parent_pid: int | None) -> None:
    """Exit when the parent process dies.

    A driver that is SIGKILLed would otherwise reparent this server, which keeps the model (and its
    GPU memory) resident indefinitely. ``parent_pid`` guards against the process having already been
    reparented before the check ran.

    A daemon thread polls the parent pid rather than arming ``PR_SET_PDEATHSIG``: the kernel delivers
    that signal when the parent *thread* that forked us exits, and the driver starts us from a thread
    pool thread, so a retired thread would SIGKILL a healthy server. Polling follows the parent
    process, which is what "the driver died" actually means, and costs one getppid per second.
    """
    if parent_pid is None:
        return

    if os.getppid() != parent_pid:
        os._exit(1)

    def watch() -> None:
        while True:
            time.sleep(1.0)
            if os.getppid() != parent_pid:
                os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    _die_with_parent(args.parent_pid)

    overrides = {
        "peak_height": args.peak_height,
        "peak_distance": args.peak_distance,
        "smoothing_sigma": args.smoothing_sigma,
        "peak_prominence": args.peak_prominence,
        "edge_margin": args.edge_margin,
    }

    try:
        detector = PageBreakDetector.from_checkpoint(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
            target_width=args.target_width,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001 - report and exit like the CLI
        print(json.dumps({"error": f"Failed to load model: {exc}"}))
        sys.exit(1)

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[union-attr]
    DetectionServer(
        detector,
        idle_cache_release=args.idle_cache_release,
    ).run()


if __name__ == "__main__":
    main()
