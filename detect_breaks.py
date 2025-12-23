#!/usr/bin/env python3
"""
CLI Page Break Detector.

This script detects page breaks in an image using a trained model and outputs
the results as JSON to stdout. It is designed to be called by other applications.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from PIL import Image

# Ensure local modules can be imported
sys.path.append(os.getcwd())

try:
    from page_break_model import DeepPageBreakDetector, gaussian_smooth, find_peaks_torch
except ImportError:
    # Fallback if running from a different directory but modules are in the same dir as script
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from page_break_model import DeepPageBreakDetector, gaussian_smooth, find_peaks_torch
    except ImportError:
        print(json.dumps({"error": "Could not import 'page_break_model'. Make sure you are in the project root or the script directory."}))
        sys.exit(1)

import torch.nn.functional as F

def predict_sliding_window(model, x, chunk_size=2048, overlap=512):
    """
    Performs inference using a sliding window approach.
    """
    b, c, h, w = x.shape
    x_padded = F.pad(x, (0, 0, overlap, overlap), mode='constant', value=0)
    h_padded = x_padded.shape[2]
    
    full_logits = torch.zeros((b, h), device=x.device)
    count_map = torch.zeros((b, h), device=x.device)
    
    for start_y in range(0, h, chunk_size):
        p_start = start_y 
        p_end = min(p_start + chunk_size + 2 * overlap, h_padded)
        
        chunk = x_padded[:, :, p_start:p_end, :]
        
        with torch.no_grad():
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

def main():
    parser = argparse.ArgumentParser(description="Detect page breaks in an image and output JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, required=True, help="Path to model config (.json)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    
    # Calibration params (defaults None to allow config fallback)
    parser.add_argument("--peak-height", type=float, default=None, help="Minimum height for a peak (0.0-1.0)")
    parser.add_argument("--peak-distance", type=int, default=None, help="Minimum distance between peaks (pixels)")
    parser.add_argument("--smoothing-sigma", type=float, default=None, help="Gaussian smoothing sigma")
    parser.add_argument("--peak-prominence", type=float, default=None, help="Minimum prominence of peaks")
    
    # Edge constraint
    parser.add_argument("--edge-margin", type=int, default=100, help="Pixels from top/bottom to ignore splits (in resized coordinates)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load config: {str(e)}"}))
        sys.exit(1)

    # Resolve Parameters (CLI > Config > Default)
    peak_height = args.peak_height if args.peak_height is not None else config.get("peak_height", 0.5)
    peak_distance = args.peak_distance if args.peak_distance is not None else config.get("peak_distance", 50)
    smoothing_sigma = args.smoothing_sigma if args.smoothing_sigma is not None else config.get("smoothing_sigma", 1.0)
    peak_prominence = args.peak_prominence if args.peak_prominence is not None else config.get("peak_prominence", 0.1)

    # 2. Load Model
    try:
        model = DeepPageBreakDetector(**config)
        state_dict = torch.load(args.checkpoint, map_location=device)
        
        # Handle Lightning prefix
        state_dict_keys = list(state_dict.keys())
        if state_dict_keys and all(k.startswith("model.") for k in state_dict_keys):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {str(e)}"}))
        sys.exit(1)

    # 3. Load and Preprocess Image
    try:
        img = Image.open(args.image).convert("RGB")
        w_orig, h_orig = img.size
        
        target_width = 768
        scale = target_width / w_orig
        new_h = int(h_orig * scale)
        img_resized = img.resize((target_width, new_h), resample=Image.Resampling.BILINEAR)
        
        img_np = np.array(img_resized)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        input_tensor = img_t.unsqueeze(0).to(device)
    except Exception as e:
        print(json.dumps({"error": f"Failed to process image: {str(e)}"}))
        sys.exit(1)

    # 4. Inference
    try:
        with torch.no_grad():
            logits = predict_sliding_window(model, input_tensor)
            probs = torch.sigmoid(logits)
            
            # Smoothing
            if smoothing_sigma > 0.01:
                probs = gaussian_smooth(probs, smoothing_sigma)
            
            # Peak Detection
            peaks = find_peaks_torch(
                probs, 
                peak_height, 
                int(peak_distance), 
                peak_prominence
            )
            
            # Edge Constraint
            if peaks.shape[1] > (args.edge_margin * 2):
                peaks[:, :args.edge_margin] = False
                peaks[:, -args.edge_margin:] = False
            
            # Get indices
            peak_indices = torch.nonzero(peaks.squeeze(0)).flatten().cpu().numpy()
            peak_probs = probs.squeeze(0)[peak_indices].cpu().numpy()
            
            # Convert back to original coordinates
            splits = []
            for y_new, prob in zip(peak_indices, peak_probs):
                y_orig = int(y_new / scale)
                splits.append({
                    "y_resized": int(y_new),
                    "y_original": y_orig,
                    "confidence": float(prob)
                })
            
            output = {
                "image": args.image,
                "original_height": h_orig,
                "original_width": w_orig,
                "resized_height": new_h,
                "scale_factor": scale,
                "parameters": {
                    "peak_height": peak_height,
                    "peak_distance": peak_distance,
                    "smoothing_sigma": smoothing_sigma,
                    "peak_prominence": peak_prominence,
                    "edge_margin": args.edge_margin
                },
                "splits": splits,
                "count": len(splits)
            }
            
            print(json.dumps(output, indent=2))

    except Exception as e:
        print(json.dumps({"error": f"Inference failed: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
