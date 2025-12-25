"""
Deep Page Break Detector Model.

This module defines the neural network architecture for detecting page breaks
in vertical manga strips, along with helper functions for Gaussian smoothing
and peak detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_optimal_device() -> torch.device:
    """
    Returns the best available device for inference.
    Prioritizes torch.accelerator > CUDA > ROCm (via CUDA) > MPS > XPU > CPU.
    """
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        if device is not None:
            return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
        
    # Check for XPU (Intel)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
        
    return torch.device("cpu")

def gaussian_smooth(waveform: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    1D Gaussian smoothing compatible with ONNX.
    """
    if sigma <= 0.01:
        return waveform
        
    # Determine kernel size (must be odd)
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Generate grid
    radius = kernel_size // 2
    x_grid = torch.arange(kernel_size, dtype=waveform.dtype, device=waveform.device) - radius
    
    # Calculate Gaussian
    kernel = torch.exp(-x_grid**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # Reshape for conv1d: (Out, In, Kernel) -> (1, 1, K)
    kernel = kernel.view(1, 1, -1)
    
    # Apply to input
    # Input expects (Batch, Length). Unsqueeze to (Batch, 1, Length)
    x_in = waveform.unsqueeze(1) 
    
    # Padding matches 'same' convolution
    x_out = F.conv1d(x_in, kernel, padding=radius)
    
    return x_out.squeeze(1)

def find_peaks_torch(x: torch.Tensor, height: float, distance: int, prominence: float = 0.0) -> torch.Tensor:
    """
    Peak detection with Height, Distance, AND Prominence support.
    Fully compatible with ONNX export.
    """
    # 1. Height threshold
    mask_height = x > height
    
    # 2. Distance & Prominence suppression
    # We use the same kernel size for both distance (max) and prominence (min) context
    k = int(distance)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    pad = k // 2
    
    # Dilation: Find local maximum (Distance check)
    x_padded = x.unsqueeze(1) # (B, 1, L)
    x_max = F.max_pool1d(x_padded, kernel_size=k, stride=1, padding=pad).squeeze(1)
    
    # Erosion: Find local minimum (Prominence check)
    # MinPool(x) is equivalent to -MaxPool(-x)
    x_min = -F.max_pool1d(-x_padded, kernel_size=k, stride=1, padding=pad).squeeze(1)
    
    # A point is a peak if:
    # A. It equals the local max (Distance constraint)
    # B. It passes the absolute height threshold
    # C. It stands out from the local background (Prominence constraint)
    
    # Calculate prominence: (Peak Value - Local Minimum Value)
    calc_prominence = x - x_min
    mask_prominence = calc_prominence > prominence
    
    # Combine conditions
    is_peak = (x == x_max) & mask_height & mask_prominence
    
    return is_peak

class ResidualWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class DeepPageBreakDetector(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64, layers=8, kernel_size=3, 
                 dropout=0.0, activation="ReLU", width_stride=4, 
                 smoothing_sigma=0.0, peak_height=0.5, peak_distance=50, peak_prominence=0.1):
        """
        Initialize a deep CNN-based page break detector.

        Parameters
        ----------
        input_channels : int, optional
            Number of input feature channels (e.g., visual/textual features per position).
        hidden_dim : int, optional
            Width of the internal convolutional feature maps.
        layers : int, optional
            Number of residual convolutional blocks.
        kernel_size : int, optional
            Convolution kernel size for the residual blocks.
        dropout : float, optional
            Dropout probability applied inside residual blocks (0.0 disables dropout).
        activation : str, optional
            Name of the activation layer from ``torch.nn`` to use (e.g., ``"ReLU"``).
        width_stride : int, optional
            Stride along the width dimension in the initial convolution, controlling
            how densely positions are sampled.
        smoothing_sigma : float, optional
            Standard deviation of the 1D Gaussian kernel applied to smooth the model's
            output scores before peak detection. A value of ``0.0`` (or very small,
            e.g., ``<= 0.01``) effectively disables smoothing. Larger values produce
            smoother, less noisy score curves but may blur sharp peaks. Typical values
            are small non-negative numbers (e.g., ``0.0``–``5.0``).
        peak_height : float, optional
            Minimum score threshold a position must exceed to be considered a candidate
            page-break peak after smoothing. Usually expected to be in the range
            ``0.0``–``1.0`` when scores represent probabilities or confidences.
            Increasing this value reduces the number of detected page breaks.
        peak_distance : int, optional
            Minimum separation between consecutive detected peaks, measured in the
            1D sequence index used for peak detection (i.e., feature steps after any
            width downsampling). Must be a positive integer; larger values enforce
            fewer, more widely spaced page-break predictions.
        peak_prominence : float, optional
            Minimum prominence (relative height) required. A peak must be at least this much
            higher than the lowest point in its `peak_distance` neighborhood.
            Helps filter out noisy ripples on top of high-confidence plateaus.
        """
        super().__init__()
        
        # Save configuration for easy export/loading
        self.config = {
            "input_channels": input_channels,
            "hidden_dim": hidden_dim,
            "layers": layers,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "activation": activation,
            "width_stride": width_stride,
            "smoothing_sigma": smoothing_sigma,
            "peak_height": peak_height,
            "peak_distance": peak_distance,
            "peak_prominence": peak_prominence
        }
        
        try:
            ActLayer = getattr(nn, activation)
        except AttributeError:
            ActLayer = nn.ReLU
        
        self.features = nn.Sequential()
        
        # Initial Convolution
        self.features.add_module("init_conv", nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, 
                      padding=kernel_size//2, stride=(1, width_stride)), 
            ActLayer(),
            nn.BatchNorm2d(hidden_dim)
        ))
        
        # Residual Blocks
        for i in range(layers):
            dilation = min(2 ** i, 1024)
            padding = dilation * (kernel_size // 2)
            
            block = nn.Sequential()
            block.add_module("conv", nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                                               padding=padding, dilation=dilation))
            block.add_module("act", ActLayer())
            block.add_module("bn", nn.BatchNorm2d(hidden_dim))
            
            if dropout > 0.0:
                block.add_module("drop", nn.Dropout2d(p=dropout))

            self.features.add_module(f"res_block_{i}", ResidualWrapper(block))

        # Classifier Head
        self.width_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.classifier = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        
        # Calibration state
        self.smoothing_sigma = float(smoothing_sigma)
        self.peak_height = float(peak_height)
        self.peak_distance = int(peak_distance)
        self.peak_prominence = float(peak_prominence)

    def forward(self, x):
        """
        Standard Forward Pass returning Logits.
        Used during training.
        """
        x = self.features(x)     
        x = self.width_pool(x)   
        x = self.classifier(x) 
        return x.squeeze(1).squeeze(-1)

    def predict(self, x, apply_calibration=True):
        """
        Run the full inference pipeline for page break detection.

        This is intended for evaluation / deployment and wraps the standard
        ``forward`` pass with a sigmoid, and optionally with smoothing and
        peak detection. During training, prefer calling ``forward`` directly
        to obtain raw logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width),
                as expected by ``forward``.
            apply_calibration (bool, optional): If True, apply the current
                calibration settings (Gaussian smoothing and peak detection)
                to the probability curve. If False, only the sigmoid is applied
                and no smoothing / peak detection is performed.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                probs: Probability curve of shape (batch, height), obtained by
                    applying a sigmoid to the logits produced by ``forward`` and,
                    if enabled, Gaussian smoothing.
                peaks: When ``apply_calibration`` is True, a boolean mask of shape
                    (batch, height) indicating detected peaks in ``probs``.
                    When ``apply_calibration`` is False, this value is ``None``.
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        
        if apply_calibration:
            # 1. Smoothing
            if self.smoothing_sigma > 0.01:
                probs = gaussian_smooth(probs, self.smoothing_sigma)
            
            # 2. Peak Detection
            peaks = find_peaks_torch(probs, self.peak_height, self.peak_distance, self.peak_prominence)
            return probs, peaks
        
        return probs, None

    def set_calibration(self, height, distance, sigma, prominence):
        """
        Updates the internal calibration parameters.
        """
        self.peak_height = float(height)
        self.peak_distance = int(distance)
        self.smoothing_sigma = float(sigma)
        self.peak_prominence = float(prominence)
        
        self.config.update({
            "peak_height": height, 
            "peak_distance": distance, 
            "smoothing_sigma": sigma,
            "peak_prominence": prominence
        })