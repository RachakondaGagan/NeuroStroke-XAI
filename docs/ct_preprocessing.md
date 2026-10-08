# CT Preprocessing Specification

## Overview

The CT imaging pipeline for **NeuroStroke-XAI** ingests head CT scans to perform 3-way classification:
- **Normal**
- **Ischemic stroke**
- **Hemorrhagic stroke**

This document details the exact preprocessing methodology, architectural decisions, and geometric constraints applied to every CT image prior to feature extraction and neural network inference.

---

## 1. Input Image Characteristics & Data Reality

From our comprehensive audit of the curated 4,200 TEKNOFEST CT dataset:
- **Data format**: 8-bit RGB JPEG images (rendered pseudo-color / colormapped CT slices).
- **Not DICOM / HU**: Images are pre-windowed and rendered RGB representations, **not** raw 12-bit or 16-bit DICOM files with Hounsfield Units (HU).
- **Color space**: 3-channel RGB. Conversion to grayscale is strictly avoided as color tinting and colormap gradations convey intensity thresholds embedded during the rendering process.
- **Dimensional distribution**:
  - `512 × 512`: 4,059 images (96.64%)
  - Variable dimensions / aspect ratios: 141 images (3.36%), such as `512 × 759`.

---

## 2. Core Preprocessing Operations

### 2.1 Target Dimensions & Resolution
- **Standard resolution**: `224 × 224` pixels.
- Compatible with standard deep learning vision backbones (ResNet, EfficientNet, ConvNeXt, DenseNet).

### 2.2 Aspect-Ratio-Preserving Resize (`ResizeAndPad`)
Medical CT images depict anatomical structures where spherical symmetry and spatial proportions (ventricles, sulci, midline) are critical diagnostic cues. Direct non-uniform resizing (`224 × 224`) would distort these structures (e.g., stretching or squishing an intracerebral hematoma or infarction).

The deterministic resizing algorithm:
1. **Aspect Ratio Preservation**:
   $$\text{scale} = \frac{\text{IMAGE\_SIZE}}{\max(\text{width}, \text{height})}$$
   $$\text{new\_width} = \text{round}(\text{width} \times \text{scale})$$
   $$\text{new\_height} = \text{round}(\text{height} \times \text{scale})$$
2. **Bilinear Resampling**:
   Resize the original image to `(new_width, new_height)` using `PIL.Image.Resampling.BILINEAR`.
3. **Centered Zero-Padding**:
   Place the resized image into the exact center of a black (`fill=0`) square canvas of size `IMAGE_SIZE × IMAGE_SIZE` (`224 × 224`):
   $$\text{left} = \lfloor (\text{IMAGE\_SIZE} - \text{new\_width}) / 2 \rfloor$$
   $$\text{top} = \lfloor (\text{IMAGE\_SIZE} - \text{new\_height}) / 2 \rfloor$$

### 2.3 Strict Transformation Exclusions
The following standard computer-vision augmentations are **explicitly prohibited**:
- ❌ **No Random Crops**: Cropping could eliminate focal ischemic lesions or marginal intracranial hemorrhages near the calvarium.
- ❌ **No Horizontal Flips**: Brain anatomy exhibits lateralized pathology. While hemispheric symmetry exists, arbitrary flipping interferes with anatomical orientation analysis and unilateral clinical correlation.
- ❌ **No Vertical Flips**: Brain anatomy has strict superior/inferior orientation; inverted scans do not correspond to clinical views.
- ❌ **No Color Jitter / Hue Alterations**: Since the dataset consists of pseudocolor renderings, altering hue/saturation shifts the pseudo-intensity mappings and obscures lesion densities.

---

## 3. Augmentation Policy (Training vs. Validation/Test)

### 3.1 Training Pipeline
To encourage model invariance to slight patient positioning differences in the scanner while preserving clinical features:
1. **`ResizeAndPad(224, fill=0)`**: Deterministic aspect-ratio preservation and padding.
2. **Conservative Rotation**: `RandomRotation(degrees=5, fill=0)`. Bounded to $\pm 5^\circ$ (simulating minor head tilt in the gantry).
3. **Conservative Affine Translation & Scaling**:
   - Translation: $\le 2\%$ in either direction (`translate=(0.02, 0.02)`).
   - Scaling: $0.98$ to $1.02$ (`scale=(0.98, 1.02)`).
4. **ToTensor**: Scales pixel values from $[0, 255]$ uint8 to $[0.0, 1.0]$ float32 tensor of shape `(3, 224, 224)`.
5. **Channel Normalization**:
   $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$

### 3.2 Validation and Test Pipeline
Strictly deterministic with **zero** stochastic augmentations:
1. **`ResizeAndPad(224, fill=0)`**
2. **ToTensor**
3. **Channel Normalization** (using identical parameters derived from the training set)

---

## 4. Normalization Strategy

- **Policy**: Normalization parameters must be calculated **exclusively on the training split** (`data/splits/train.csv`, $N = 2,947$) to prevent any statistical leakage into validation and test sets.
- **Reference ImageNet statistics** (used as initial baseline fallback):
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- **Dataset-Specific Training Statistics**:
  Calculated across all $2,947$ training images transformed via `ResizeAndPad(224)` and stored in `data/inspection/training_normalization_stats.json`.

---

## 5. Implementation Architecture & Modularity

- Preprocessing transformations are defined in [`src/preprocessing/transforms.py`](file:///Users/gagan/Desktop/NeuroStroke-XAI/src/preprocessing/transforms.py).
- The `ResizeAndPad` class is implemented purely with standard Python and PIL (`Pillow`), allowing visual audits and preprocessing verification without requiring heavyweight deep-learning frameworks.
- Framework-specific transforms (e.g. PyTorch `transforms.Compose`) are conditionally exposed when `torchvision` is available.
