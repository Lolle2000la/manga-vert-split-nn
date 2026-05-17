# Manga Vertical Split Neural Network

A specialized, local-first deep learning solution for detecting page breaks (splits) in manga when the pages are given in a concatenated format. This project provides a complete, reproducible pipeline from dataset generation to training, optimization, and deployment-ready inference.

| Manga Split Visualized | With Raw Logits |
|-----------------------|-------------|
| ![Example of detected splits on a manga strip](assets/manga%20splits%20visualized.png) | ![Logits output showing peaks at split locations](assets/manga%20splits%20visualized%20with%20raw%20logits.png) |

## Overview & Purpose

When manga pages are scanned or stored digitally, they are often combined into long vertical strips. Automatically detecting where to split these strips back into individual pages is a non-trivial task, especially when scene transitions are subtle or when there is minimal whitespace between pages.

This project was built from the ground up as a **Local-First** solution. Processing large libraries of high-resolution images demands significant bandwidth and raises privacy concerns if sent to a cloud API. By running locally, this model ensures your entire library stays private, functions completely offline, and processes images at maximum speed without network bottlenecks.

## Model Architecture (WaveNet-style 1D/2D CNN)

Rather than relying on a generic ResNet or Vision Transformer, this network implements a custom architecture specifically designed for vertical image analysis, heavily inspired by **WaveNet**.

1.  **Exponentially Dilated Convolutions**: The core residual blocks use exponentially increasing dilation factors ($1, 2, 4, 8, 16...$). This grants the network a massive receptive field—allowing it to "see" far up and down the strip to understand context like speech bubbles or panel borders—without pooling, thereby perfectly preserving vertical pixel resolution.
2.  **Symmetric Context**: Unlike the original causal WaveNet, this model uses symmetric padding (`padding = dilation * (kernel_size // 2)`). This allows the network to look both "into the past" (above the split) and "into the future" (below the split) simultaneously, which is critical for visual scene transitions.
3.  **Residual Connections**: The deep structure (e.g., 8+ layers) uses residual connections to prevent vanishing gradients and allow additive feature learning.
4.  **1D Probability Mapping**: The 2D visual context is elegantly mapped into a 1D probability distribution using an `AdaptiveAvgPool2d((None, 1))` over the horizontal axis, drastically reducing compute while retaining the crucial vertical predictions.

The output is then calibrated using a custom post-processing peak detection algorithm (configurable via height, distance, and prominence constraints) optimized for the F-Beta score.

## Quick Start & Reproducibility

This repository utilizes `uv` for lightning-fast, perfectly reproducible dependency management.

### Prerequisites
*   Python 3.12 (Aim currently requires Python 3.12)
*   `uv` package manager

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Replicate the exact environment
uv sync

# 3. Activate the environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### High-Performance Inference

We provide an ONNX export script (`export_onnx.py`) to decouple the model from Python. This allows for native, high-performance execution on Linux (e.g., CachyOS) and hardware acceleration (like ROCm) for blazing-fast batch processing.

#### 1. GUI Visualizer (Python)
Interactive tool to visualize splits, probabilities, and tune parameters in real-time. This provides the visual confirmation seen in the screenshots above.

```bash
python gui_visualizer.py
```

#### 2. CLI Detection Tool (Python)
Command-line script for batch processing or integration into other pipelines. Outputs structured JSON ([schema](detect_breaks_schema.json)).

```bash
python detect_breaks.py \
  --checkpoint "models/BCE Only (v8)/final_deployment/best_model.pth" \
  --config "models/BCE Only (v8)/final_deployment/model_config.json" \
  --image "path/to/strip.jpg"
```

## Model Variants

During development, two main training strategies were explored:

1.  **BCE Only (v8) [PRODUCTION READY]**:
    *   Trained primarily with **Binary Cross Entropy** loss.
    *   **Performance**: This variant proved to be the most robust in real-world scenarios, producing sharper peaks and fewer false positives in complex scene transitions.
    *   **Artifacts**: The final deployment files (`best_model.pth`, the exported `.onnx` version, and `model_config.json`) located in `models/BCE Only (v8)/final_deployment/` represent this model. Use this for all real-world applications.
    *   **Note on Configurations**: Every trained model is accompanied by a `model_config.json` file. This file is critical as it contains not only the architectural hyperparameters (like hidden dimension and layers) but also the optimized calibration settings (peak height, distance, smoothing) discovered during the calibration phase. Always load the config alongside the checkpoint!

2.  **BCE + EMD**:
    *   Trained with a combination of BCE and **Earth Mover's Distance (Wasserstein)** loss.
    *   **Purpose**: EMD was added to explicitly penalize the "distance" of the predicted split from the ground truth, theoretically allowing the network to learn from "near misses".
    *   **Reality**: While mathematically interesting, it tended to produce overly smoothed probability distributions that were harder to calibrate for precise, pixel-perfect cutting.

## End-to-End ML Pipeline

This repository covers the entire ML lifecycle. If you wish to retrain or extend the model, follow this pipeline:

1.  **Dataset Generation**: `python generate_strips.py`
    *   Converts raw CBZ/Zip manga chapters into training strips.
    *   Stitches pages and generates Gaussian-smoothed 1D target labels.
2.  **Training**: `python page_break_trainer.py --data_dir dataset_strips --epochs 50`
    *   Trains the model using PyTorch Lightning.
3.  **Hyperparameter Optimization**: `python optimize.py --data_dir dataset_strips --n_trials 50`
    *   Uses Optuna to find the best architecture (layers, hidden dim) and training params.
4.  **Calibration**: `python calibrate.py --mode optimize --checkpoint ...`
    *   Optimizes the post-processing parameters (Peak Height, Distance, Smoothing) on the validation set to maximize the F-Beta score.
5.  **Export**: `python export_onnx.py --checkpoint ... --config ...`
    *   Exports the model to ONNX for production deployment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.