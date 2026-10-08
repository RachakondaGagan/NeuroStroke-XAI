from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = Path("data/metadata/teknofest_3class_manifest.csv")
OUTPUT_DIR = Path("data/inspection")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(CSV_PATH)

print(f"Total images: {len(df)}")


# ============================================================
# Analyze RGB channels
# ============================================================

channel_differences = []

for _, row in df.iterrows():

    image_path = Path(row["image_path"])

    image = np.array(Image.open(image_path).convert("RGB"))

    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    rg_diff = np.mean(np.abs(r - g))
    gb_diff = np.mean(np.abs(g - b))
    rb_diff = np.mean(np.abs(r - b))

    channel_differences.append(
        [rg_diff, gb_diff, rb_diff]
    )


channel_differences = np.array(channel_differences)

print("\n" + "=" * 60)
print("RGB CHANNEL ANALYSIS")
print("=" * 60)

print(
    f"Mean |R-G|: {channel_differences[:, 0].mean():.4f}"
)

print(
    f"Mean |G-B|: {channel_differences[:, 1].mean():.4f}"
)

print(
    f"Mean |R-B|: {channel_differences[:, 2].mean():.4f}"
)


# ============================================================
# Sample images
# ============================================================

classes = ["normal", "ischemic", "hemorrhagic"]

fig, axes = plt.subplots(
    3,
    5,
    figsize=(15, 9)
)

for row_idx, class_name in enumerate(classes):

    class_df = df[df["class"] == class_name]

    samples = class_df.sample(
        n=5,
        random_state=42
    )

    for col_idx, (_, row) in enumerate(samples.iterrows()):

        image_path = Path(row["image_path"])

        image = Image.open(image_path).convert("RGB")

        axes[row_idx, col_idx].imshow(image)

        axes[row_idx, col_idx].set_title(
            f"{class_name}\n{image.size}"
        )

        axes[row_idx, col_idx].axis("off")


plt.tight_layout()

output_path = OUTPUT_DIR / "ct_sample_grid.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nSample grid saved to:")
print(output_path)


# ============================================================
# Analyze image dimensions
# ============================================================

dimension_counts = {}

for _, row in df.iterrows():

    image_path = Path(row["image_path"])

    with Image.open(image_path) as image:

        size = image.size

    dimension_counts[size] = (
        dimension_counts.get(size, 0) + 1
    )


print("\n" + "=" * 60)
print("IMAGE DIMENSIONS")
print("=" * 60)

for size, count in sorted(
    dimension_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]:

    print(f"{size}: {count}")


# ============================================================
# Analyze image intensity
# ============================================================

intensities = []

for _, row in df.iterrows():

    image_path = Path(row["image_path"])

    image = np.array(
        Image.open(image_path).convert("L")
    )

    intensities.append(
        [
            image.mean(),
            image.std(),
            image.min(),
            image.max()
        ]
    )


intensities = np.array(intensities)

print("\n" + "=" * 60)
print("IMAGE INTENSITY STATISTICS")
print("=" * 60)

print(
    f"Mean pixel intensity: "
    f"{intensities[:, 0].mean():.2f}"
)

print(
    f"Mean pixel standard deviation: "
    f"{intensities[:, 1].mean():.2f}"
)

print(
    f"Average minimum pixel value: "
    f"{intensities[:, 2].mean():.2f}"
)

print(
    f"Average maximum pixel value: "
    f"{intensities[:, 3].mean():.2f}"
)
