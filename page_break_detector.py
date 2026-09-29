"""
Reusable page-break detection logic.

This module holds everything the CLI (``detect_breaks.py``) and the long-lived
NDJSON server (``detect_server.py``) share:

* :func:`resolve_parameters` merges calibration defaults, the model config and
  explicit overrides;
* :func:`predict_sliding_window` runs the tiled inference;
* :class:`PageBreakDetector` owns a loaded model and turns an image into the
  JSON-serializable result described by ``detect_breaks_schema.json``.

The model can be injected, which keeps the class unit-testable without a real
checkpoint. :meth:`PageBreakDetector.from_checkpoint` wires up the checkpoint and
``model_config.json`` exactly like the CLI used to.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Import the model regardless of the caller's working directory: the scripts are
# executed from ``backend/src/manga-vert-split-nn`` by the host application.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.append(_MODULE_DIR)

from page_break_model import (
    DeepPageBreakDetector,
    find_peaks_torch,
    gaussian_smooth,
    get_optimal_device,
)

DEFAULT_TARGET_WIDTH = 768
DEFAULT_EDGE_MARGIN = 100
DEFAULT_CHUNK_SIZE = 2048
DEFAULT_OVERLAP = 512

# Fallbacks used when neither the CLI nor the model config provides a value.
DEFAULT_PARAMETERS: dict[str, float | int] = {
    "peak_height": 0.5,
    "peak_distance": 50,
    "smoothing_sigma": 1.0,
    "peak_prominence": 0.1,
    "edge_margin": DEFAULT_EDGE_MARGIN,
}


class PageBreakDetectionError(RuntimeError):
    """Raised when an image cannot be loaded or inference fails."""


def resolve_parameters(
    config: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, float | int]:
    """Merge calibration defaults, the model config and explicit overrides.

    ``None`` override values fall through to the config (and then the default),
    which lets the CLI leave a flag unset and still honour the config file.
    """
    config = config or {}
    overrides = overrides or {}
    resolved: dict[str, float | int] = {}
    for key, default in DEFAULT_PARAMETERS.items():
        override = overrides.get(key)
        if override is not None:
            resolved[key] = override
        else:
            resolved[key] = config.get(key, default)

    # Normalise types to keep the output schema stable.
    resolved["peak_height"] = float(resolved["peak_height"])
    resolved["peak_distance"] = int(resolved["peak_distance"])
    resolved["smoothing_sigma"] = float(resolved["smoothing_sigma"])
    resolved["peak_prominence"] = float(resolved["peak_prominence"])
    resolved["edge_margin"] = int(resolved["edge_margin"])
    return resolved


def load_model_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def build_model(config: Mapping[str, Any]) -> DeepPageBreakDetector:
    """Construct the network from a ``model_config.json`` mapping.

    Calibration keys are accepted by the constructor; unknown keys are dropped so
    a config that also carries detector-only settings still loads.
    """
    import inspect

    allowed = set(inspect.signature(DeepPageBreakDetector.__init__).parameters)
    allowed.discard("self")
    model_kwargs = {k: v for k, v in config.items() if k in allowed}
    return DeepPageBreakDetector(**model_kwargs)


def load_state_dict(checkpoint_path: str, device: torch.device) -> Mapping[str, Any]:
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Handle the Lightning ``model.`` prefix used by the training checkpoints.
    keys = list(state_dict.keys())
    if keys and all(k.startswith("model.") for k in keys):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}
    return state_dict


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return get_optimal_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def release_accelerator_cache(device: torch.device | None = None) -> None:
    """Return cached accelerator blocks to the driver (best effort).

    Used for GPU co-tenancy: the upscaling worker and this detector share the
    GPU, and releasing the caching allocator's blocks while idle lets the other
    process use the VRAM without either side reloading its model.
    """
    try:
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            torch.accelerator.empty_cache()
            return
    except Exception:  # noqa: BLE001 - best effort, never fail a request over this
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception:  # noqa: BLE001
        return


def predict_sliding_window(
    model: torch.nn.Module,
    x: torch.Tensor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    should_abort: Callable[[], bool] | None = None,
) -> torch.Tensor:
    """Run the model over a tall image in overlapping vertical windows.

    ``should_abort`` is polled between windows so a long inference can be
    cancelled between chunks (used by the NDJSON server's ``cancel`` request).
    """
    b, _c, h, _w = x.shape
    x_padded = F.pad(x, (0, 0, overlap, overlap), mode="constant", value=0)
    h_padded = x_padded.shape[2]

    full_logits = torch.zeros((b, h), device=x.device)
    count_map = torch.zeros((b, h), device=x.device)

    for start_y in range(0, h, chunk_size):
        if should_abort is not None and should_abort():
            raise PageBreakDetectionError("cancelled")

        p_start = start_y
        p_end = min(p_start + chunk_size + 2 * overlap, h_padded)

        chunk = x_padded[:, :, p_start:p_end, :]

        chunk_logits = model(chunk)

        l_out = chunk_logits.shape[1]
        global_start = start_y - overlap
        global_end = global_start + l_out

        valid_start = max(0, global_start)
        valid_end = min(h, global_end)

        chunk_start = valid_start - global_start
        chunk_end = chunk_start + (valid_end - valid_start)

        full_logits[:, valid_start:valid_end] += chunk_logits[:, chunk_start:chunk_end]
        count_map[:, valid_start:valid_end] += 1.0

    mask = count_map > 0
    full_logits[mask] /= count_map[mask]

    return full_logits


class PageBreakDetector:
    """Turns an image into the page-break result described by the JSON schema.

    The model is expected to be already loaded, in ``eval`` mode and on the
    target device. Use :meth:`from_checkpoint` for the usual path.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        parameters: Mapping[str, float | int] | None = None,
        device: torch.device | None = None,
        target_width: int = DEFAULT_TARGET_WIDTH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        self.model = model
        self.parameters = resolve_parameters(None, parameters)
        self.device = device or get_optimal_device()
        self.target_width = int(target_width)
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str | None = None,
        config: Mapping[str, Any] | None = None,
        device: str | torch.device | None = None,
        target_width: int = DEFAULT_TARGET_WIDTH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        overrides: Mapping[str, Any] | None = None,
    ) -> PageBreakDetector:
        if config is None:
            if config_path is None:
                raise ValueError("Either config_path or config must be provided.")
            config = load_model_config(config_path)

        resolved_device = resolve_device(device)
        model = build_model(config)
        state_dict = load_state_dict(checkpoint_path, resolved_device)
        model.load_state_dict(state_dict)
        model.to(resolved_device)
        model.eval()

        return cls(
            model=model,
            parameters=resolve_parameters(config, overrides),
            device=resolved_device,
            target_width=target_width,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    @property
    def edge_margin(self) -> int:
        return int(self.parameters["edge_margin"])

    def _prepare_tensor(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, int, int, float, int]:
        width_orig, height_orig = image.size
        scale = self.target_width / width_orig
        new_h = int(height_orig * scale)
        resized = image.resize(
            (self.target_width, new_h), resample=Image.Resampling.BILINEAR
        )

        image_np = np.array(resized)
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        input_tensor = image_t.unsqueeze(0).to(self.device)
        return input_tensor, width_orig, height_orig, scale, new_h

    def _infer(
        self,
        input_tensor: torch.Tensor,
        scale: float,
        height_orig: int,
        width_orig: int,
        new_h: int,
        image_name: str,
        should_abort: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        params = self.parameters
        edge_margin = int(params["edge_margin"])

        with torch.inference_mode():
            logits = predict_sliding_window(
                self.model,
                input_tensor,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                should_abort=should_abort,
            )
            probs = torch.sigmoid(logits)

            if float(params["smoothing_sigma"]) > 0.01:
                probs = gaussian_smooth(probs, float(params["smoothing_sigma"]))

            peaks = find_peaks_torch(
                probs,
                float(params["peak_height"]),
                int(params["peak_distance"]),
                float(params["peak_prominence"]),
            )

            if edge_margin > 0 and peaks.shape[1] > (edge_margin * 2):
                peaks[:, :edge_margin] = False
                peaks[:, -edge_margin:] = False

            peak_indices = torch.nonzero(peaks.squeeze(0)).flatten().cpu().numpy()
            peak_probs = probs.squeeze(0)[peak_indices].cpu().numpy()

            splits = []
            for y_new, prob in zip(peak_indices, peak_probs):
                y_orig = int(y_new / scale)
                splits.append(
                    {
                        "y_resized": int(y_new),
                        "y_original": y_orig,
                        "confidence": float(prob),
                    }
                )

        return {
            "image": image_name,
            "original_height": int(height_orig),
            "original_width": int(width_orig),
            "resized_height": int(new_h),
            "scale_factor": float(scale),
            "parameters": {
                "peak_height": float(params["peak_height"]),
                "peak_distance": int(params["peak_distance"]),
                "smoothing_sigma": float(params["smoothing_sigma"]),
                "peak_prominence": float(params["peak_prominence"]),
                "edge_margin": edge_margin,
            },
            "splits": splits,
            "count": len(splits),
        }

    def detect_image(
        self,
        image: Image.Image,
        image_name: str = "",
        should_abort: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        try:
            input_tensor, width_orig, height_orig, scale, new_h = self._prepare_tensor(
                image
            )
        except Exception as exc:
            raise PageBreakDetectionError(f"Failed to process image: {exc}") from exc

        try:
            return self._infer(
                input_tensor,
                scale,
                height_orig,
                width_orig,
                new_h,
                image_name,
                should_abort,
            )
        except PageBreakDetectionError:
            raise
        except Exception as exc:
            raise PageBreakDetectionError(f"Inference failed: {exc}") from exc

    def detect_path(
        self,
        image_path: str,
        should_abort: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise PageBreakDetectionError(f"Failed to process image: {exc}") from exc

        try:
            return self.detect_image(image, image_path, should_abort)
        finally:
            image.close()

    def detect_bytes(
        self,
        data: bytes,
        image_name: str = "",
        should_abort: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        import io

        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise PageBreakDetectionError(f"Failed to process image: {exc}") from exc

        try:
            return self.detect_image(image, image_name, should_abort)
        finally:
            image.close()
