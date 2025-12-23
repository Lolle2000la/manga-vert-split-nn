#!/usr/bin/env python3
"""
Calibration script for the Page Break Detector.

This script optimizes the post-processing parameters (peak height, distance,
prominence, smoothing) using Optuna to maximize the Soft F-Beta score.
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import optuna
import torch
from optuna.samplers import GPSampler
from tqdm import tqdm

from page_break_model import DeepPageBreakDetector, gaussian_smooth, find_peaks_torch
from page_break_trainer import StripDataset, DataLoader, collate_variable_height


# --- METRIC (Soft F-Beta) ---
def calculate_soft_fbeta(
    pred_indices_list: List[torch.Tensor],
    gt_indices_list: List[torch.Tensor],
    tolerance: int = 30,
    beta: float = 1.0
) -> float:
    """
    Calculates Soft F-Beta Score.

    Args:
        pred_indices_list: List of predicted peak indices for each sample.
        gt_indices_list: List of ground truth peak indices for each sample.
        tolerance: Maximum distance to consider a match.
        beta: Beta value for F-score.
              beta < 1.0: Prioritizes Precision (Penalizes wrong splits more).
              beta > 1.0: Prioritizes Recall (Penalizes missed splits more).

    Returns:
        The calculated Soft F-Beta score.
    """
    tp = 0
    fp = 0
    fn = 0
    soft_tp = 0.0
    
    for p_inds, t_inds in zip(pred_indices_list, gt_indices_list):
        n_p = len(p_inds)
        n_t = len(t_inds)
        
        if n_t == 0:
            fp += n_p
            continue
        if n_p == 0:
            fn += n_t
            continue
            
        # Ensure floating point for distance calc
        p_inds = p_inds.float()
        t_inds = t_inds.float()
            
        dists = torch.abs(p_inds.unsqueeze(1) - t_inds.unsqueeze(0))
        min_dists, gt_indices = torch.min(dists, dim=1)
        valid_mask = min_dists <= tolerance
        
        sorted_inds = torch.argsort(min_dists)
        matched_gt = torch.zeros(n_t, dtype=torch.bool)
        
        sample_tp = 0
        sample_soft_tp = 0.0
        
        for idx in sorted_inds:
            if not valid_mask[idx]:
                continue
            
            gt_idx = gt_indices[idx]
            if not matched_gt[gt_idx]:
                matched_gt[gt_idx] = True
                sample_tp += 1
                
                dist = min_dists[idx].item()
                quality = max(0.0, 1.0 - (dist / tolerance))
                sample_soft_tp += quality
        
        tp += sample_tp
        soft_tp += sample_soft_tp
        fp += (n_p - sample_tp)
        fn += (n_t - sample_tp)
        
    epsilon = 1e-6
    soft_precision = soft_tp / (tp + fp + epsilon)
    soft_recall = soft_tp / (tp + fn + epsilon)
    
    # F-Beta Formula
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (soft_precision * soft_recall) / ((beta_sq * soft_precision) + soft_recall + epsilon)
    
    return f_beta

# --- CACHE STEP ---
def load_standalone_model(checkpoint_path: str, config_path: Optional[str] = None) -> DeepPageBreakDetector:
    """
    Loads the model from a checkpoint and config file.

    Args:
        checkpoint_path: Path to the model checkpoint (.pth).
        config_path: Path to the model config (.json). If None, looks in the same directory as checkpoint.

    Returns:
        The loaded DeepPageBreakDetector model.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    model = DeepPageBreakDetector(**config)
    
    state_dict = torch.load(checkpoint_path, map_location='cuda')
    
    state_dict_keys = list(state_dict.keys())
    if state_dict_keys and all(k.startswith("model.") for k in state_dict_keys):
        prefix_len = len("model.")
        state_dict = {k[prefix_len:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval().cuda()
    return model

def cache_predictions(args):
    model = load_standalone_model(args.checkpoint, args.config)
    
    ds = StripDataset(args.data_dir, crop_height=None, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_variable_height)
    
    all_preds = []
    all_targets = []
    
    print(f"[Info] Inference on {len(ds)} samples (Batch={args.batch_size})...")
    
    with torch.no_grad():
        for x, y in tqdm(dl):
            x = x.cuda()
            logits = model(x)
            probs = torch.sigmoid(logits).cpu()
            
            all_preds.append(probs)
            all_targets.append(y.cpu())
            
    torch.save({"preds": all_preds, "targets": all_targets}, args.cache_file)
    print(f"[Info] Cache saved to {args.cache_file}")

# --- OPTIMIZE STEP ---
def run_optimization(args):
    if not os.path.exists(args.cache_file):
        print(f"Error: {args.cache_file} not found.")
        sys.exit(1)

    print("[Info] Loading Cache...")
    data = torch.load(args.cache_file)
    
    preds_list = []
    targets_list = []
    
    for batch_p, batch_t in zip(data["preds"], data["targets"]):
        for i in range(batch_p.shape[0]):
            preds_list.append(batch_p[i].float().unsqueeze(0))
            targets_list.append(batch_t[i].float().unsqueeze(0))

    print("[Info] Pre-calculating Ground Truth Indices...")
    gt_indices_list = []
    
    for t in targets_list:
        gt_peaks = find_peaks_torch(t, height=0.5, distance=10)
        indices = torch.nonzero(gt_peaks[0]).squeeze(1)
        gt_indices_list.append(indices)

    def objective(trial):
        height = trial.suggest_float("peak_height", 0.1, 0.95)
        distance = trial.suggest_int("peak_distance", 10, 1000)
        sigma = trial.suggest_float("smoothing_sigma", 0.0, 20.0)
        prominence = trial.suggest_float("peak_prominence", 0.0, 0.95) 
        
        pred_indices_list = []
        for raw_probs in preds_list:
            if sigma > 0.01:
                smoothed = gaussian_smooth(raw_probs, sigma)
            else:
                smoothed = raw_probs
            
            is_peak = find_peaks_torch(smoothed, height, distance, prominence)
            
            inds = torch.nonzero(is_peak[0]).squeeze(1)
            pred_indices_list.append(inds)
            
        return calculate_soft_fbeta(pred_indices_list, gt_indices_list, tolerance=30, beta=args.beta)
    print(f"[Info] Study: {args.study_name} | Target Beta: {args.beta}")

    sampler = GPSampler(seed=42, deterministic_objective=True)

    study = optuna.create_study(
        direction="maximize", 
        storage=args.storage, 
        study_name=args.study_name, 
        load_if_exists=True,
        sampler=sampler
    )
    
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\n=== CALIBRATION COMPLETE ===")
    print(f"Best Soft F-{args.beta}: {study.best_value:.4f}")
    print("Best Params:", study.best_params)
    
    if args.config:
        print(f"[Info] Updating config file {args.config} with best params...")
        with open(args.config, 'r') as f:
            conf = json.load(f)
        conf.update(study.best_params)
        with open(args.config, 'w') as f:
            json.dump(conf, f, indent=4)
        print("Updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize post-processing parameters (peak detection) for the Page Break Detector.")
    parser.add_argument("--mode", choices=["cache", "optimize"], required=True, help="Operation mode: 'cache' to pre-calculate model predictions, 'optimize' to run Optuna search.")
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint (.pth) to use for calibration.")
    parser.add_argument("--config", type=str, help="Path to the model configuration file (.json).")
    parser.add_argument("--data_dir", type=str, help="Path to the dataset directory containing 'val' or 'test' folders.")
    parser.add_argument("--cache_file", type=str, default="calibration_cache.pt", help="File path to save/load cached predictions (default: calibration_cache.pt).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference during caching (default: 4).")
    parser.add_argument("--storage", type=str, default="sqlite:///calibration.db", help="Optuna storage URL (default: sqlite:///calibration.db).")
    parser.add_argument("--study_name", type=str, default="calibration_study", help="Name of the Optuna study (default: calibration_study).")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials to run (default: 100).")
    parser.add_argument("--beta", type=float, default=1.0, 
                        help="F-Beta score. < 1.0 prefers Precision (fewer wrong breaks), > 1.0 prefers Recall.")
    
    args = parser.parse_args()
    
    if args.mode == "cache":
        cache_predictions(args)
    elif args.mode == "optimize":
        run_optimization(args)