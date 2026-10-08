# CT Baseline Experiment Plan (Experiment 01)

## 1. Experiment Overview

- **Experiment ID**: `EXP-CT-01-BASELINE`
- **Goal**: Establish the first reliable, reproducible 3-class baseline model for brain CT stroke detection (Normal vs. Ischemic vs. Hemorrhagic) on the leakage-free group-level splits.
- **Model**: Pretrained ResNet-50 (ImageNet-1k weights).
- **Status**: Ready for execution once the PyTorch/torchvision execution runtime is verified.

---

## 2. Experimental Configuration & Hyperparameters

```yaml
experiment:
  name: ct_3class_resnet50_baseline
  task: multiclass_classification
  num_classes: 3
  seed: 42

dataset:
  train_split: data/splits/train.csv (2,947 images)
  val_split: data/splits/validation.csv (621 images)
  test_split: data/splits/test.csv (632 images)
  classes:
    0: normal
    1: ischemic
    2: hemorrhagic

preprocessing:
  input_size: [224, 224]
  color_mode: RGB
  resize_method: ResizeAndPad (aspect-ratio preserving)
  interpolation: bilinear
  fill_color: 0 (black padding)
  train_augmentation:
    rotation_degrees: 5
    affine_translation: [0.02, 0.02]
    affine_scale: [0.98, 1.02]
  val_test_augmentation: none (deterministic)
  normalization:
    mean: [0.485, 0.456, 0.406]  # ImageNet initial; update to training-derived if finalized
    std: [0.229, 0.224, 0.225]

training:
  backbone: resnet50
  pretrained: true
  freeze_backbone_epochs: 3  # Warmup head only
  unfreeze_all_epochs: 22    # Total epochs = 25
  batch_size: 32
  optimizer: AdamW
  learning_rate:
    backbone: 1.0e-4
    head: 1.0e-3
  weight_decay: 1.0e-2
  scheduler: CosineAnnealingLR
  min_lr: 1.0e-6
  loss: CrossEntropyLoss
  mixed_precision: true (if CUDA/MPS supported)

evaluation:
  monitor_metric: val_macro_f1
  mode: max
  patience: 8
  save_best_only: true
```

---

## 3. Step-by-Step Execution Protocol

### Step A: Dataset & DataLoader Instantiation
- Instantiate `CTStrokeDataset` for train, validation, and test splits.
- Verify zero sample overlap between DataLoader iterators.
- Set DataLoader `shuffle=True` for train, `shuffle=False` for validation/test.
- Use `num_workers=4` (or 2 depending on host CPU) and `pin_memory=True`.

### Step B: Two-Stage Fine-Tuning
1. **Stage 1 (Head Warmup, Epochs 1–3)**:
   - Freeze all convolutional backbone weights (`requires_grad = False`).
   - Train only the new linear classifier `fc = nn.Linear(2048, 3)` with learning rate $10^{-3}$.
   - Stabilizes the random projection layer before backpropagating through pre-trained feature extractors.
2. **Stage 2 (End-to-End Fine-Tuning, Epochs 4–25)**:
   - Unfreeze all layers (`requires_grad = True`).
   - Reduce backbone learning rate to $10^{-4}$ while keeping head at $5 \times 10^{-4}$.
   - Apply cosine annealing schedule.

### Step C: Validation & Early Stopping
- At the end of each epoch, evaluate on the validation set:
  - Validation Loss
  - Validation Accuracy
  - Validation Macro F1
- Checkpoint whenever `val_macro_f1 > best_val_f1`.
- If no improvement for 8 consecutive epochs, trigger early stopping.

### Step D: Final Independent Test Set Evaluation
- Load the checkpoint with highest validation Macro F1.
- Run complete evaluation on the untouched `data/splits/test.csv` ($N = 632$):
  - Generate full classification report (precision, recall, F1 per class).
  - Compute normalized confusion matrix.
  - Record per-case aggregate predictions.

---

## 4. Expected Deliverables & Artifacts

1. Checkpoint file: `models/checkpoints/ct_baseline_resnet50_best.pt`
2. Metrics summary: `experiments/results/ct_baseline_resnet50_metrics.json`
3. Visual artifacts:
   - Training & validation loss/accuracy curves (`experiments/results/ct_baseline_curves.png`)
   - Test set confusion matrix (`experiments/results/ct_baseline_confusion_matrix.png`)
4. Verification that model outputs are formatted properly for subsequent **Grad-CAM** integration in Phase 4.
