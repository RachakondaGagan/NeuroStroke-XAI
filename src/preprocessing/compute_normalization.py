"""
Compute RGB Mean and Standard Deviation on the Training Split.

This script processes all images in the training set (data/splits/train.csv)
using the exact deterministic ResizeAndPad(224) transform and calculates the dataset-specific
RGB channel mean and standard deviation using pure PIL and NumPy.

The computed statistics are saved to:
    data/inspection/training_normalization_stats.json
"""

import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from transforms import ResizeAndPad, IMAGE_SIZE

TRAIN_SPLIT_PATH = PROJECT_ROOT / "data/splits/train.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/inspection/training_normalization_stats.json"


def compute_training_stats():
    if not TRAIN_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Training split not found at: {TRAIN_SPLIT_PATH}")

    # Read training paths using standard csv module (fast and zero overhead)
    image_paths = []
    with open(TRAIN_SPLIT_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = Path(row["image_path"])
            abs_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
            image_paths.append(abs_path)

    total_images = len(image_paths)
    print(f"Loaded training split: {total_images} images.")
    assert total_images == 2947, f"Expected 2947 training images, found {total_images}"

    resize_pad = ResizeAndPad(IMAGE_SIZE)

    sum_rgb = np.zeros(3, dtype=np.float64)
    sum_sq_rgb = np.zeros(3, dtype=np.float64)
    pixels_per_image = IMAGE_SIZE * IMAGE_SIZE
    total_pixels = float(total_images * pixels_per_image)

    print("Computing RGB channel statistics across training images...")
    for idx, img_path in enumerate(image_paths):
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            transformed = resize_pad(img_rgb)
            # Ensure shape is exactly (224, 224, 3)
            assert transformed.size == (IMAGE_SIZE, IMAGE_SIZE), (
                f"Unexpected size {transformed.size} for image {img_path}"
            )
            # Normalize pixel values to [0.0, 1.0]
            arr = np.asarray(transformed, dtype=np.float64) / 255.0

            # Accumulate sum and sum of squares across spatial dimensions (H, W)
            sum_rgb += arr.sum(axis=(0, 1))
            sum_sq_rgb += (arr ** 2).sum(axis=(0, 1))

        if (idx + 1) % 500 == 0 or (idx + 1) == total_images:
            print(f"Processed {idx + 1}/{total_images} images ({(idx + 1) / total_images * 100:.1f}%)...")

    mean_rgb = (sum_rgb / total_pixels).tolist()
    variance_rgb = (sum_sq_rgb / total_pixels) - (np.array(mean_rgb) ** 2)
    # Ensure numerical stability
    variance_rgb = np.maximum(variance_rgb, 0.0)
    std_rgb = np.sqrt(variance_rgb).tolist()

    stats = {
        "dataset": "CT Kaggle 3-Class Curated",
        "split": "train",
        "csv_source": "data/splits/train.csv",
        "num_images": total_images,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "calculation_method": (
            "Exact single-pass accumulation of pixel sums and squared pixel sums "
            "over all 2,947 training images preprocessed with deterministic ResizeAndPad(224) "
            "in RGB color space normalized to [0.0, 1.0]. Zero stochastic augmentations applied. "
            "Validation and test sets strictly excluded."
        ),
        "rgb_mean": [round(float(m), 6) for m in mean_rgb],
        "rgb_std": [round(float(s), 6) for s in std_rgb],
        "channel_order": ["R", "G", "B"]
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print("\n" + "=" * 60)
    print("Computed Training Normalization Statistics:")
    print(f"  Total Images: {stats['num_images']}")
    print(f"  Image Size:   {stats['image_size']}")
    print(f"  RGB Mean:     {stats['rgb_mean']}")
    print(f"  RGB Std:      {stats['rgb_std']}")
    print(f"Saved to:       {OUTPUT_PATH}")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    compute_training_stats()
