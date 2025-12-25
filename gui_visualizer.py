#!/usr/bin/env python3
"""
GUI Visualizer for Page Break Detector.

This script provides a minimal GUI to load a trained model, load an image,
and visualize the page break predictions, including probabilities and split lines.
It allows interactive adjustment of post-processing parameters.
"""

import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore
from PIL import Image

# Ensure local modules can be imported
sys.path.append(os.getcwd())

try:
    from page_break_model import DeepPageBreakDetector, gaussian_smooth, find_peaks_torch, get_optimal_device
except ImportError:
    print("Error: Could not import 'page_break_model'. Make sure you are in the project root.")
    sys.exit(1)


def predict_sliding_window(model, x, chunk_size=2048, overlap=512):
    """
    Performs inference using a sliding window approach to handle large images
    without running out of memory (OOM) and to match training consistency.
    """
    b, c, h, w = x.shape
    # Pad input to handle edges and overlaps
    # Pad format is (left, right, top, bottom)
    # We pad top and bottom with 'overlap' amount
    x_padded = F.pad(x, (0, 0, overlap, overlap), mode='constant', value=0)
    
    h_padded = x_padded.shape[2]
    
    # List to store output chunks
    # We will reconstruct the full logit map
    full_logits = torch.zeros((b, h), device=x.device)
    count_map = torch.zeros((b, h), device=x.device)
    
    # Slide over height
    # We step by chunk_size
    for start_y in range(0, h, chunk_size):
        # The window includes the overlap context
        # Window start in padded coords
        p_start = start_y 
        p_end = min(p_start + chunk_size + 2 * overlap, h_padded)
        
        chunk = x_padded[:, :, p_start:p_end, :]
        
        # Run model
        with torch.no_grad():
            chunk_logits = model(chunk) # (B, L_chunk)
            
        # Determine valid region in the output (removing the overlap context)
        # The model output corresponds to p_start to p_end
        
        # We want to map this back to the original 'x' coordinates (0 to h)
        # The current chunk covers x[start_y - overlap : start_y + chunk_size + overlap]
        # But we only want to keep the center part for the final map, 
        # or we can average the overlaps.
        
        # Simple approach: Linear blending or just add and divide by count
        
        # Length of this output chunk
        l_out = chunk_logits.shape[1]
        
        # Map to global unpadded coordinates
        # The chunk starts at 'p_start' in padded space.
        # 'p_start' corresponds to 'start_y - overlap' in unpadded space.
        global_start = start_y - overlap
        global_end = global_start + l_out
        
        # Clip to valid unpadded range [0, h]
        valid_start = max(0, global_start)
        valid_end = min(h, global_end)
        
        # Indices within the chunk to take
        chunk_start = valid_start - global_start
        chunk_end = chunk_start + (valid_end - valid_start)
        
        # Accumulate
        full_logits[:, valid_start:valid_end] += chunk_logits[:, chunk_start:chunk_end]
        count_map[:, valid_start:valid_end] += 1.0

    # Average overlapping regions
    # Avoid division by zero
    mask = count_map > 0
    full_logits[mask] /= count_map[mask]
    
    return full_logits


class PageBreakVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Page Break Detector Visualizer")
        self.root.geometry("1200x900")

        self.model: Optional[DeepPageBreakDetector] = None
        self.device = get_optimal_device()
        self.current_image_path = None
        self.original_image = None  # PIL Image
        self.processed_tensor = None # Tensor (1, C, H, W)
        self.logits = None # Tensor (1, H)
        self.probs = None # Tensor (1, H)
        self.peaks = None # Tensor (1, H)
        
        self.debounce_timer = None

        # --- GUI Layout ---
        # Left Panel: Controls
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Right Panel: Visualization
        self.vis_frame = ttk.Frame(root, padding="10")
        self.vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._setup_controls()
        self._setup_visualization()

        # Initial Status
        self.status_var.set(f"Device: {self.device}")

    def _setup_controls(self):
        # Model Loading
        lbl_model = ttk.Label(self.control_frame, text="Model Configuration", font=("Arial", 12, "bold"))
        lbl_model.pack(pady=(0, 5), anchor="w")

        btn_load_model = ttk.Button(self.control_frame, text="Load Model (Config/PTH)", command=self.load_model)
        btn_load_model.pack(fill=tk.X, pady=5)

        self.lbl_model_status = ttk.Label(self.control_frame, text="No model loaded", foreground="red")
        self.lbl_model_status.pack(anchor="w", pady=2)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Image Loading
        lbl_img = ttk.Label(self.control_frame, text="Input Image", font=("Arial", 12, "bold"))
        lbl_img.pack(pady=(0, 5), anchor="w")

        btn_load_img = ttk.Button(self.control_frame, text="Load Image", command=self.load_image)
        btn_load_img.pack(fill=tk.X, pady=5)

        self.lbl_img_status = ttk.Label(self.control_frame, text="No image loaded", foreground="gray")
        self.lbl_img_status.pack(anchor="w", pady=2)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Parameters
        lbl_params = ttk.Label(self.control_frame, text="Calibration Parameters", font=("Arial", 12, "bold"))
        lbl_params.pack(pady=(0, 5), anchor="w")

        # Sliders
        self.param_vars = {
            "peak_height": tk.DoubleVar(value=0.5),
            "peak_distance": tk.IntVar(value=50),
            "smoothing_sigma": tk.DoubleVar(value=1.0),
            "peak_prominence": tk.DoubleVar(value=0.1)
        }

        self._create_slider("Peak Height", "peak_height", 0.0, 1.0, 0.01)
        self._create_slider("Min Distance", "peak_distance", 1, 2000, 10)
        self._create_slider("Smoothing Sigma", "smoothing_sigma", 0.0, 20.0, 0.1)
        self._create_slider("Prominence", "peak_prominence", 0.0, 1.0, 0.01)

        btn_update = ttk.Button(self.control_frame, text="Update / Re-run", command=self.run_inference)
        btn_update.pack(fill=tk.X, pady=10)

        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Visualization Options
        lbl_opts = ttk.Label(self.control_frame, text="View Options", font=("Arial", 12, "bold"))
        lbl_opts.pack(pady=(0, 5), anchor="w")

        self.show_lines = tk.BooleanVar(value=True)
        self.show_probs = tk.BooleanVar(value=True)
        self.show_logits = tk.BooleanVar(value=False)
        self.show_peaks = tk.BooleanVar(value=True)

        ttk.Checkbutton(self.control_frame, text="Show Split Lines", variable=self.show_lines, command=self.update_plot).pack(anchor="w")
        ttk.Checkbutton(self.control_frame, text="Show Probability", variable=self.show_probs, command=self.update_plot).pack(anchor="w")
        ttk.Checkbutton(self.control_frame, text="Show Raw Logits", variable=self.show_logits, command=self.update_plot).pack(anchor="w")
        ttk.Checkbutton(self.control_frame, text="Show Peak Markers", variable=self.show_peaks, command=self.update_plot).pack(anchor="w")

        # Status Bar
        self.status_var = tk.StringVar()
        lbl_status = ttk.Label(self.control_frame, textvariable=self.status_var, wraplength=200)
        lbl_status.pack(side=tk.BOTTOM, anchor="w", pady=10)

    def _create_slider(self, label, var_name, min_val, max_val, step):
        frame = ttk.Frame(self.control_frame)
        frame.pack(fill=tk.X, pady=2)
        
        lbl = ttk.Label(frame, text=label)
        lbl.pack(anchor="w")
        
        scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=self.param_vars[var_name], command=lambda v: self._on_slider_change())
        scale.pack(fill=tk.X)
        
        val_lbl = ttk.Label(frame, textvariable=self.param_vars[var_name])
        val_lbl.pack(anchor="e")

    def _on_slider_change(self):
        if self.debounce_timer is not None:
            self.root.after_cancel(self.debounce_timer)
        self.debounce_timer = self.root.after(500, self.run_inference)

    def _setup_visualization(self):
        # Matplotlib Figure
        self.fig = Figure(figsize=(8, 10), dpi=100)
        # 1 row, 2 cols: Image, Probability Plot
        self.ax_img = self.fig.add_subplot(121)
        self.ax_plot = self.fig.add_subplot(122, sharey=self.ax_img)
        
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.vis_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.vis_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model Config or Checkpoint",
            filetypes=[("JSON Config", "*.json"), ("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        config_path = None
        checkpoint_path = None

        if file_path.endswith(".json"):
            config_path = file_path
            # Try to find checkpoint in same dir
            base_dir = os.path.dirname(file_path)
            possible_ckpt = os.path.join(base_dir, "best_model.pth")
            if os.path.exists(possible_ckpt):
                checkpoint_path = possible_ckpt
            else:
                # Ask for checkpoint
                checkpoint_path = filedialog.askopenfilename(title="Select Checkpoint (.pth)", filetypes=[("PyTorch Checkpoint", "*.pth")])
        elif file_path.endswith(".pth"):
            checkpoint_path = file_path
            # Try to find config
            base_dir = os.path.dirname(file_path)
            possible_config = os.path.join(base_dir, "model_config.json")
            if os.path.exists(possible_config):
                config_path = possible_config
            else:
                config_path = filedialog.askopenfilename(title="Select Config (.json)", filetypes=[("JSON Config", "*.json")])

        if not config_path or not checkpoint_path:
            messagebox.showerror("Error", "Both Config and Checkpoint are required.")
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update sliders with config defaults if available
            if 'peak_height' in config: self.param_vars['peak_height'].set(config['peak_height'])
            if 'peak_distance' in config: self.param_vars['peak_distance'].set(config['peak_distance'])
            if 'smoothing_sigma' in config: self.param_vars['smoothing_sigma'].set(config['smoothing_sigma'])
            if 'peak_prominence' in config: self.param_vars['peak_prominence'].set(config['peak_prominence'])

            self.model = DeepPageBreakDetector(**config)
            
            # Load weights
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle Lightning prefix
            state_dict_keys = list(state_dict.keys())
            if state_dict_keys and all(k.startswith("model.") for k in state_dict_keys):
                state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.lbl_model_status.config(text=f"Loaded: {os.path.basename(checkpoint_path)}", foreground="green")
            self.status_var.set("Model loaded successfully.")
            
            # If image is already loaded, re-run
            if self.current_image_path:
                self.run_inference()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            print(e)

    def load_image(self):
        # Cleanup previous image data to prevent OOM
        if self.processed_tensor is not None:
            del self.processed_tensor
            self.processed_tensor = None
        
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            torch.accelerator.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.current_image_path = file_path
            img = Image.open(file_path).convert("RGB")
            
            # Resize logic similar to training
            target_width = 768
            w, h = img.size
            scale = target_width / w
            new_h = int(h * scale)
            img_resized = img.resize((target_width, new_h), resample=Image.Resampling.BILINEAR)
            
            self.original_image = img_resized
            
            # Prepare tensor
            img_np = np.array(img_resized)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            self.processed_tensor = img_t.unsqueeze(0).to(self.device) # (1, 3, H, W)

            self.lbl_img_status.config(text=f"Loaded: {os.path.basename(file_path)} ({new_h}px height)")
            
            if self.model:
                self.run_inference()
            else:
                # Just show image
                self.update_plot(only_image=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            print(e)

    def run_inference(self):
        if not self.model or self.processed_tensor is None:
            return

        # Update model calibration parameters
        self.model.set_calibration(
            height=self.param_vars['peak_height'].get(),
            distance=int(self.param_vars['peak_distance'].get()),
            sigma=self.param_vars['smoothing_sigma'].get(),
            prominence=self.param_vars['peak_prominence'].get()
        )

        try:
            with torch.no_grad():
                # Use sliding window inference for consistency and OOM safety
                logits = predict_sliding_window(self.model, self.processed_tensor) # (B, L)
                self.logits = logits.cpu().numpy()[0]
                
                probs = torch.sigmoid(logits)
                
                # 1. Smoothing
                sigma = self.param_vars['smoothing_sigma'].get()
                if sigma > 0.01:
                    probs = gaussian_smooth(probs, sigma)
                
                self.probs = probs.cpu().numpy()[0]
                
                # 2. Peak Detection
                peaks = find_peaks_torch(
                    probs, 
                    self.param_vars['peak_height'].get(), 
                    int(self.param_vars['peak_distance'].get()), 
                    self.param_vars['peak_prominence'].get()
                )
                
                # Constraint: No splits within 100px of top/bottom
                if peaks.shape[1] > 200:
                    peaks[:, :100] = False
                    peaks[:, -100:] = False
                
                self.peaks = peaks.cpu().numpy()[0]
                
            self.update_plot()
            
            # Count splits
            num_splits = np.sum(self.peaks)
            self.status_var.set(f"Inference Complete. Found {num_splits} splits.")

        except Exception as e:
            messagebox.showerror("Error", f"Inference failed:\n{e}")
            print(e)

    def update_plot(self, only_image=False):
        if self.original_image is None:
            return

        self.ax_img.clear()
        self.ax_plot.clear()

        # 1. Show Image
        # Matplotlib displays images with (0,0) at top-left by default
        self.ax_img.imshow(self.original_image)
        self.ax_img.set_title("Input Image")
        self.ax_img.axis('off')

        if only_image:
            self.canvas.draw()
            return

        height = self.original_image.height
        y_axis = np.arange(height)

        # 2. Show Probability / Logits
        if self.probs is not None:
            # We want plot to be vertical: x=prob, y=height
            # Invert y axis to match image (0 at top)
            
            if self.show_logits.get() and self.logits is not None:
                # Plot raw logits (unscaled)
                self.ax_plot.plot(self.logits, y_axis, label="Logits", color="green", linewidth=1, alpha=0.5)
            
            if self.show_probs.get():
                self.ax_plot.plot(self.probs, y_axis, label="Probability", color="blue", linewidth=1)
                # Add threshold line
                thresh = self.param_vars['peak_height'].get()
                self.ax_plot.axvline(x=thresh, color='gray', linestyle='--', alpha=0.5, label="Threshold")

            # Set limits based on what's shown
            if self.show_probs.get() and not self.show_logits.get():
                self.ax_plot.set_xlim(0, 1.1)
            
            self.ax_plot.set_ylim(height, 0) # Invert Y
            self.ax_plot.grid(True, alpha=0.3)
            self.ax_plot.set_title("Model Output")

        # 3. Show Peaks / Splits
        if self.peaks is not None:
            peak_indices = np.where(self.peaks)[0]
            
            if self.show_lines.get():
                for y in peak_indices:
                    self.ax_img.axhline(y=y, color='red', linestyle='-', linewidth=2, alpha=0.7)
            
            if self.show_peaks.get():
                # Mark on the plot
                if len(peak_indices) > 0 and self.probs is not None:
                    peak_probs = self.probs[peak_indices]
                    self.ax_plot.scatter(peak_probs, peak_indices, color='red', marker='x', s=50, label="Split Point", zorder=5)

        self.ax_plot.legend(loc='upper right')
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = PageBreakVisualizerApp(root)
    root.mainloop()
