# Manga Vertical Split Neural Network

A specialized deep learning solution for detecting page breaks (splits) in manga when the pages are given in a concatenated format. This project provides a complete pipeline from dataset generation to training, optimization, and deployment-ready inference tools.

## Overview

When manga pages are scanned or stored digitally, they are often combined into long vertical strips. Automatically detecting where to split these strips back into individual pages is a non-trivial task, especially when scene transitions are subtle or when there is minimal whitespace between pages.

This project implements a **ResNet-style Convolutional Neural Network**, that analyzes the visual content of a strip and predicts the probability of a page break at every pixel row. For that it maps the input image to a 1D probability distribution over its height, indicating likely split points.

The model parameters include calibration settings for peak detection, which are already optimized for best F-Beta score (with Beta here being 0.5) on a test set.

### Key Features
*   **Deep Learning Approach**: Uses a 1D CNN to understand context, superior to simple whitespace detection.
*   **Robust Inference**: Sliding window inference handles images of any height without memory issues.
*   **Post-Processing Calibration**: Advanced peak detection with configurable height, distance, and prominence constraints.
*   **Complete Toolset**: Includes GUI visualizer, CLI tool, training scripts, and ONNX export.

## Quick Start

### Prerequisites
*   Python 3.12 (Aim currently requires Python 3.12)
*   `uv` package manager

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Create a virtual environment
uv venv --python 3.12

# 3. Activate the environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 4. Install dependencies
uv pip install -e .
```

### Inference (Using the Pre-trained Model)

We provide two main tools for using the trained model:

#### 1. GUI Visualizer
Interactive tool to visualize splits, probabilities, and tune parameters in real-time.

```bash
python gui_visualizer.py
```
*   Load the model config/checkpoint.
*   Load an image.
*   Adjust sliders to see how sensitivity affects split detection.

#### 2. CLI Detection Tool
Command-line script for batch processing or integration into other applications (e.g., C# pipelines). Outputs JSON.

```bash
python detect_breaks.py \
  --checkpoint "models/BCE Only (v8)/final_deployment/best_model.pth" \
  --config "models/BCE Only (v8)/final_deployment/model_config.json" \
  --image "path/to/strip.jpg"
```

**Output Format:**
```json
{
  "splits": [
    { "y_original": 1953, "confidence": 0.98 },
    ...
  ],
  "count": 5
}
```

## Model Variants

During development, two main training strategies were explored:

1.  **BCE Only (v8) [RECOMMENDED]**:
    *   Trained primarily with **Binary Cross Entropy** loss.
    *   **Performance**: This variant proved to be more robust in real-world scenarios, producing sharper peaks and fewer false positives in complex scene transitions.
    *   **Use this for production.**

2.  **BCE + EMD**:
    *   Trained with a combination of BCE and **Earth Mover's Distance (Wasserstein)** loss.
    *   **Theory**: EMD attempts to account for the "distance" of the predicted split from the ground truth, theoretically allowing for "near misses".
    *   **Reality**: While mathematically interesting, it tended to produce overly smoothed probability distributions that were harder to calibrate for precise cutting.

## Development Pipeline

If you wish to retrain or extend the model, follow this pipeline:

### 1. Dataset Generation
Convert raw CBZ/Zip manga chapters into training strips with ground truth labels.
```bash
python generate_strips.py
```
*   Scans a directory of CBZ files.
*   Stitches pages together and generates vertical strips.
*   Creates Gaussian-smoothed target labels for training.

### 2. Training
Train the model using PyTorch Lightning.
```bash
python page_break_trainer.py --data_dir dataset_strips --epochs 50
```

### 3. Hyperparameter Optimization
Use Optuna to find the best architecture (layers, hidden dim) and training params.
```bash
python optimize.py --data_dir dataset_strips --n_trials 50
```

### 4. Calibration
Once a model is trained, optimize the post-processing parameters (Peak Height, Distance, Smoothing) to maximize the F-Beta score.
```bash
python calibrate.py --mode optimize --checkpoint ...
```

### 5. Export
Export the model to ONNX for high-performance inference in non-Python environments.
```bash
python export_onnx.py --checkpoint ... --config ...
```

## Project Structure

*   `page_break_model.py`: Core PyTorch model definition (1D ResNet).
*   `page_break_trainer.py`: Training loop and data loading.
*   `gui_visualizer.py`: Tkinter-based GUI for testing models.
*   `detect_breaks.py`: CLI script for JSON output.
*   `calibrate.py`: Script to tune peak detection thresholds.
*   `optimize.py`: Optuna script for architecture search.
*   `generate_strips.py`: Data preprocessing script.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.