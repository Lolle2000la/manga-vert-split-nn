#!/usr/bin/env python3
"""
ONNX Export Script for Page Break Detector.

This script exports the trained PyTorch model to ONNX format, including
the post-processing logic (calibration) if wrapped correctly.
"""

import argparse
import json
from typing import Tuple

import torch
from page_break_model import DeepPageBreakDetector

class OnnxWrapper(torch.nn.Module):
    """
    Wrapper to export the 'predict' method logic.
    We return both the probabilities and the boolean peak mask.
    """
    def __init__(self, core_model: DeepPageBreakDetector):
        super().__init__()
        self.model = core_model
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # We explicitly call predict with calibration enabled
        probs, peaks = self.model.predict(x, apply_calibration=True)
        # Ensure peaks is not None for ONNX export (it shouldn't be with apply_calibration=True)
        if peaks is None:
            peaks = torch.zeros_like(probs, dtype=torch.bool)
        return probs, peaks

def export_model(args: argparse.Namespace) -> None:
    # 1. Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    print(f"[Info] Initializing model with calibration params: "
          f"H={config.get('peak_height')}, D={config.get('peak_distance')}, "
          f"S={config.get('smoothing_sigma')}, P={config.get('peak_prominence')}")

    # 2. Init Model
    model = DeepPageBreakDetector(**config)
    
    # 3. Load Weights
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    
    # Remove Lightning prefix if present
    state_dict_keys = list(state_dict.keys())
    if state_dict_keys and all(k.startswith("model.") for k in state_dict_keys):
        prefix_len = len("model.")
        state_dict = {k[prefix_len:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. Dummy Input 
    # (Batch size 1, 3 channels, Height 2048, Width 768)
    # The height can be dynamic in ONNX, but we provide a standard shape for tracing
    x = torch.randn(1, 3, 2048, 768)
    
    # 5. Export
    wrapper = OnnxWrapper(model)
    
    print(f"[Info] Exporting to {args.output}...")
    
    # We use opset_version 17 to ensure support for operators like MaxPool1d/Conv1d with dynamic shapes
    # We disable dynamo to avoid strict constraint violations with dynamic shapes in this specific model architecture
    
    # Prepare dynamic shapes for dynamo export
    from torch.export import Dim
    batch_dim = Dim("batch_size", min=1)
    height_dim = Dim("height", min=384) # Minimum height to support convolutions
    
    # Input x is (Batch, 3, Height, 768)
    dynamic_shapes = ({0: batch_dim, 2: height_dim},)

    torch.onnx.export(
        wrapper,
        (x,),
        args.output,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['probabilities', 'peak_mask'],
        dynamo=True,
        dynamic_shapes=dynamic_shapes
    )
    print("[Success] Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the trained PyTorch model to ONNX format.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.json (must include calibration params).")
    parser.add_argument("--output", type=str, default="page_break_detector.onnx", help="Output path for the ONNX model (default: page_break_detector.onnx).")
    args = parser.parse_args()
    export_model(args)