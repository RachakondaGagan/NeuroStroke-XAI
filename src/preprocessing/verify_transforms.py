from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import sys

# Ensure module import works whether run from root or src/preprocessing
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from transforms import ResizeAndPad, IMAGE_SIZE


# ============================================================
# Configuration
# ============================================================

SPLIT_PATH = PROJECT_ROOT / "data/splits/train.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load training split
# ============================================================

df = pd.read_csv(SPLIT_PATH)

print(f"Training images available: {len(df)}")


# ============================================================
# Select representative images
# ============================================================

samples = []

classes = [
    "normal",
    "ischemic",
    "hemorrhagic"
]


# ------------------------------------------------------------
# One representative image from each class
# ------------------------------------------------------------

for class_name in classes:

    class_df = df[df["class"] == class_name]

    sample = class_df.sample(
        n=1,
        random_state=42
    ).iloc[0]

    samples.append(sample)


# ------------------------------------------------------------
# Find one variable-dimension image
#
# Inspect images to find one whose dimensions != (512, 512).
# ------------------------------------------------------------

variable_sample = None

for _, row in df.iterrows():

    raw_path = Path(row["image_path"])
    image_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path

    with Image.open(image_path) as image:

        if image.size != (512, 512):

            variable_sample = row
            break


if variable_sample is not None:
    samples.append(variable_sample)


# ============================================================
# Transformation
# ============================================================

resize_pad = ResizeAndPad(IMAGE_SIZE)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    len(samples),
    2,
    figsize=(10, 5 * len(samples))
)

if len(samples) == 1:
    axes = [axes]


# ============================================================
# Process selected images
# ============================================================

for row_idx, sample in enumerate(samples):

    raw_path = Path(sample["image_path"])
    image_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path

    class_name = sample["class"]

    original = Image.open(
        image_path
    ).convert("RGB")

    transformed = resize_pad(original)

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[row_idx][0].imshow(original)

    axes[row_idx][0].set_title(
        f"Original\n"
        f"{class_name} | {original.size}"
    )

    axes[row_idx][0].axis("off")

    # --------------------------------------------------------
    # Transformed
    # --------------------------------------------------------

    axes[row_idx][1].imshow(transformed)

    axes[row_idx][1].set_title(
        f"Resize + Padding\n"
        f"{transformed.size}"
    )

    axes[row_idx][1].axis("off")

    print(
        f"{class_name}: "
        f"{original.size} → {transformed.size}"
    )


plt.tight_layout()


# ============================================================
# Save
# ============================================================

output_path = (
    OUTPUT_DIR /
    "preprocessing_verification.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Final validation
# ============================================================

for sample in samples:

    raw_path = Path(sample["image_path"])
    image_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path

    original = Image.open(
        image_path
    ).convert("RGB")

    transformed = resize_pad(original)

    assert transformed.size == (
        IMAGE_SIZE,
        IMAGE_SIZE
    )


print()
print("✓ All selected images transformed successfully.")
print("✓ All transformed images are 224 × 224.")
print()
print("Verification image:")
print(output_path)