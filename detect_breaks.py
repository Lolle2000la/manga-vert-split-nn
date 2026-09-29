#!/usr/bin/env python3
"""
CLI Page Break Detector.

This script detects page breaks in an image using a trained model and outputs
the results as JSON to stdout. It is designed to be called by other applications.

For repeated calls (e.g. one process per chapter) prefer ``detect_server.py``,
which keeps the model resident instead of reloading it for every image.
"""

import argparse
import json
import sys

from page_break_detector import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TARGET_WIDTH,
    PageBreakDetectionError,
    PageBreakDetector,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect page breaks in an image and output JSON."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config (.json)"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")

    # Calibration params (defaults None to allow config fallback)
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

    # Edge constraint
    parser.add_argument(
        "--edge-margin",
        type=int,
        default=None,
        help="Pixels from top/bottom to ignore splits (in resized coordinates)",
    )

    # Inference knobs (rarely changed)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
    except Exception as exc:  # noqa: BLE001 - report as JSON like the old CLI
        print(json.dumps({"error": f"Failed to load model: {exc}"}))
        sys.exit(1)

    try:
        output = detector.detect_path(args.image)
    except PageBreakDetectionError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
