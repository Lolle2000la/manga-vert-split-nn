#!/usr/bin/env python3
"""
Dataset Generation Script.

This script processes CBZ manga files, extracts images, and generates
vertical strips for training the Page Break Detector.
"""

import os
import random
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = "dataset_strips"
SOURCE_DIR = "/path/to/your/data"

# Split Ratios (Applied to SERIES, not individual files)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Strip Constraints
STRIP_LENGTH_MIN = 3000
STRIP_LENGTH_MAX = 15000
GAUSSIAN_SIGMA = 5

# Aspect Ratio Filtering (Strict Single Page)
MIN_ASPECT_RATIO = 1.1
MAX_ASPECT_RATIO = 2.0


def load_image_from_cbz(zip_file: zipfile.ZipFile, img_name: str) -> Optional[np.ndarray]:
    """Reads an image from a zip file into a numpy array."""
    try:
        with zip_file.open(img_name) as file:
            img_data = file.read()
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
    except Exception:
        return None


def generate_gaussian_target(length: int, breaks: List[int], sigma: int = 5) -> np.ndarray:
    """Generates the 1D Gaussian signal for page breaks."""
    target = np.zeros(length, dtype=np.float32)
    valid_breaks = [b for b in breaks if 0 <= b < length]
    if not valid_breaks:
        return target

    target[valid_breaks] = 1.0
    target = gaussian_filter1d(target, sigma=sigma, mode='constant', cval=0.0)

    if target.max() > 0:
        target = target / target.max()

    return target


def process_chapter(cbz_path: str, output_base: Path) -> None:
    """
    Process a single CBZ and save strips to the specific output_base
    (e.g., dataset_strips/train)
    """
    try:
        with zipfile.ZipFile(cbz_path, 'r') as z:
            all_files = sorted(z.namelist())
            img_files = [f for f in all_files if f.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.webp'))]

            if not img_files:
                return

            loaded_imgs = []
            widths = []

            # 1. Load and Filter
            for f in img_files:
                img = load_image_from_cbz(z, f)
                if img is None:
                    continue

                h, w = img.shape[:2]
                if w == 0:
                    continue

                aspect_ratio = h / w
                if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
                    loaded_imgs.append(img)
                    widths.append(w)

            if not loaded_imgs:
                return

            # 2. Normalize Width
            target_width = int(np.median(widths))

            current_strip = []
            current_height = 0
            break_indices = []
            strip_counter = 0
            cbz_name = Path(cbz_path).stem

            # 3. Build Strips
            for img in loaded_imgs:
                h, w = img.shape[:2]

                scale = target_width / w
                new_h = int(h * scale)
                img_resized = cv2.resize(
                    img, (target_width, new_h), interpolation=cv2.INTER_AREA)

                # Check if strip is full
                if current_height + new_h > STRIP_LENGTH_MAX and current_height > STRIP_LENGTH_MIN:
                    full_strip = np.vstack(current_strip)
                    target_1d = generate_gaussian_target(
                        current_height, break_indices, sigma=GAUSSIAN_SIGMA)

                    filename = f"{cbz_name}_strip_{strip_counter}"
                    cv2.imwrite(str(output_base / "images" /
                                f"{filename}.jpg"), full_strip)
                    np.save(str(output_base / "labels" /
                            f"{filename}.npy"), target_1d)

                    current_strip = []
                    current_height = 0
                    break_indices = []
                    strip_counter += 1

                current_strip.append(img_resized)
                if current_height > 0:
                    break_indices.append(current_height)

                current_height += new_h

            # Save remainder
            if current_strip and current_height > STRIP_LENGTH_MIN:
                full_strip = np.vstack(current_strip)
                target_1d = generate_gaussian_target(
                    current_height, break_indices, sigma=GAUSSIAN_SIGMA)
                filename = f"{cbz_name}_strip_{strip_counter}"
                cv2.imwrite(str(output_base / "images" /
                            f"{filename}.jpg"), full_strip)
                np.save(str(output_base / "labels" /
                        f"{filename}.npy"), target_1d)

    except zipfile.BadZipFile:
        print(f"Skipping bad zip: {cbz_path}")


def main():
    source = Path(SOURCE_DIR)
    root_out = Path(OUTPUT_DIR)

    if not source.exists():
        print(f"Error: Source directory {source} does not exist.")
        return

    # 1. Discovery & Grouping
    print("Scanning for series...")
    all_cbzs = list(source.rglob('*.cbz'))

    # Dictionary mapping "Series Name" -> [List of CBZ Paths]
    series_map = {}
    for cbz in all_cbzs:
        # Assuming folder structure: /root/SeriesName/Vol1.cbz
        # cbz.parent.name gives "SeriesName"
        series_name = cbz.parent.name

        if series_name not in series_map:
            series_map[series_name] = []
        series_map[series_name].append(cbz)

    series_names = list(series_map.keys())
    random.shuffle(series_names)

    total_series = len(series_names)
    print(f"Found {len(all_cbzs)} files across {total_series} unique series.")

    # 2. Split by SERIES
    train_end = int(total_series * TRAIN_RATIO)
    val_end = train_end + int(total_series * VAL_RATIO)

    train_series = series_names[:train_end]
    val_series = series_names[train_end:val_end]
    test_series = series_names[val_end:]

    # Flatten back to file lists
    splits = {
        'train': [f for s in train_series for f in series_map[s]],
        'val':   [f for s in val_series for f in series_map[s]],
        'test':  [f for s in test_series for f in series_map[s]]
    }

    print(
        f"Split (Series): Train={len(train_series)}, Val={len(val_series)}, Test={len(test_series)}")
    print(
        f"Split (Files):  Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # 3. Process
    for split_name, files in splits.items():
        split_dir = root_out / split_name
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

        print(f"Processing {split_name} set...")
        for cbz in tqdm(files, desc=split_name):
            process_chapter(cbz, split_dir)


if __name__ == "__main__":
    main()
