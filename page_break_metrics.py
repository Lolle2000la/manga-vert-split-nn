"""
Metrics for Page Break Detection.

This module defines custom metrics for evaluating the performance of the
page break detector, specifically the Soft F-Beta score and peak detection logic.
"""

import torch
import torch.nn.functional as F
from torchmetrics import Metric

class PageBreakMetrics(Metric):
    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor
    soft_tp: torch.Tensor
    total_distance: torch.Tensor
    distance_count: torch.Tensor

    def __init__(self, dist_sync_on_step=False, tolerance=30, peak_height=0.5, peak_distance=50):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tolerance = tolerance
        self.peak_height = peak_height
        self.peak_distance = peak_distance

        # Accumulators
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum") 
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("soft_tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("distance_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _find_peaks_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        GPU-friendly peak detection using MaxPool1d as a dilation filter.
        Approximates scipy.signal.find_peaks(distance=...).
        """
        # 1. Height threshold
        mask_height = x > self.peak_height
        
        # 2. Distance suppression (approximate using MaxPool)
        # We pad to keep dimensions same. kernel_size ~ distance.
        k = self.peak_distance
        if k % 2 == 0: k += 1 # Ensure odd for symmetric padding
        pad = k // 2
        
        # This dilation creates a "plateau" around local maxes
        x_max = F.max_pool1d(x.unsqueeze(1), kernel_size=k, stride=1, padding=pad).squeeze(1)
        
        # A point is a peak if it equals the local max (and passes height threshold)
        # Note: We use a small epsilon for float comparison safety
        is_peak = (x == x_max) & mask_height
        
        return is_peak

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Keep everything on GPU!
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.sigmoid(preds)

        batch_size = preds.shape[0]
        
        # 1. Find peaks in parallel for the whole batch
        pred_mask = self._find_peaks_torch(preds)
        
        # Ground truth peaks (GT usually is binary, but let's be safe)
        # We can treat non-zero GT as peaks, or use the same finder if GT is fuzzy
        true_mask = self._find_peaks_torch(target) if target.dtype.is_floating_point else (target > 0.5)

        # 2. Iterate batch (cheap on GPU for small batch sizes) 
        # Vectorizing the matching logic across variable numbers of peaks per sample is complex,
        # so we loop over the batch, but use Tensor ops inside.
        for i in range(batch_size):
            p_indices = torch.nonzero(pred_mask[i]).squeeze(1).float()
            t_indices = torch.nonzero(true_mask[i]).squeeze(1).float()
            
            n_p = p_indices.numel()
            n_t = t_indices.numel()

            if n_t == 0:
                self.fp += n_p
                continue
            
            if n_p == 0:
                self.fn += n_t
                continue

            # 3. Distance Matrix: |P - T|
            # (N_p, 1) - (1, N_t) -> (N_p, N_t)
            dists = torch.abs(p_indices.unsqueeze(1) - t_indices.unsqueeze(0))
            
            # Find closest GT for each Prediction
            min_dists, gt_indices = torch.min(dists, dim=1) # Values, Indices of closest GT
            
            # Filter by tolerance
            valid_match_mask = min_dists <= self.tolerance
            
            # Handle double-assignment: A GT peak can only be claimed once.
            # We prioritize closer matches.
            # Simple greedy strategy on GPU:
            matched_gt = torch.zeros(n_t, dtype=torch.bool, device=self.device)
            
            # Sort matches by distance to be greedy
            sorted_indices = torch.argsort(min_dists)
            
            batch_tp = 0.0
            batch_soft_tp = 0.0
            
            for idx in sorted_indices:
                if not valid_match_mask[idx]:
                    continue
                    
                gt_idx = gt_indices[idx]
                if not matched_gt[gt_idx]:
                    # We have a match!
                    matched_gt[gt_idx] = True
                    batch_tp += 1.0
                    
                    # Soft Score
                    dist = min_dists[idx]
                    quality = torch.clamp(1.0 - (dist / self.tolerance), min=0.0)
                    batch_soft_tp += quality
                    
                    self.total_distance += dist
                    self.distance_count += 1
            
            self.tp += batch_tp
            self.soft_tp += batch_soft_tp
            self.fp += (n_p - batch_tp)
            self.fn += (n_t - matched_gt.sum())

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        soft_precision = self.soft_tp / (self.tp + self.fp + 1e-6)
        soft_recall = self.soft_tp / (self.tp + self.fn + 1e-6)
        soft_f1 = 2 * (soft_precision * soft_recall) / (soft_precision + soft_recall + 1e-6)

        if self.distance_count == 0:
            avg_pixel_error = torch.tensor(-1.0, device=self.device)
        else:
            avg_pixel_error = self.total_distance / self.distance_count

        return {
            "val_f1": f1,
            "val_soft_f1": soft_f1, 
            "val_precision": precision,
            "val_recall": recall,
            "val_avg_pixel_error": avg_pixel_error
        }