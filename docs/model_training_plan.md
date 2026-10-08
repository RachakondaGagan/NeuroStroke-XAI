# CT 3-Class Model Training Plan

## 1. Objective

Train a baseline deep learning vision classifier to accurately distinguish between:
- **Normal CT**
- **Ischemic stroke**
- **Hemorrhagic stroke**

The model must provide calibrated probabilistic predictions suitable for integration into the clinical decision-support engine and subsequent interpretability with Grad-CAM and SHAP.

---

## 2. Model Architecture Candidates

To establish a solid, reproducible baseline before fine-tuning complex ensembles, we prioritize well-characterized convolutional backbones with strong transfer learning properties:

1. **Primary Baseline Candidate: ResNet-50 / ResNet-34**
   - *Rationale*: Residual networks are standard in medical imaging literature, have predictable gradient flow, and integrate seamlessly with Grad-CAM targeting the final residual bottleneck layer (`layer4`).
   - *Input*: Pretrained ImageNet weights (`weights='DEFAULT'`), replacing the final classification head with a linear layer `Linear(in_features, 3)`.
2. **Alternative Baseline Candidate: EfficientNet-B0 / EfficientNet-B2**
   - *Rationale*: Parameter-efficient with compound scaling; lower computational overhead while maintaining competitive classification accuracy.
3. **Alternative Baseline Candidate: ConvNeXt-Tiny**
   - *Rationale*: Modern modernized convolutional architecture offering transformer-like receptive fields without requiring quadratic attention computation.

---

## 3. Input Specification & Class Mapping

### 3.1 Input Tensor Format
- Spatial dimensions: `224 × 224` pixels.
- Channels: 3 channels (RGB).
- Tensor shape: `(B, 3, 224, 224)` where $B$ is batch size.
- Dtype: `torch.float32`.

### 3.2 Label Encoding & Mapping
To prevent silent class mismatch across training, evaluation, and production inference, the numerical encoding is strictly fixed as:

| Class Name | Label Index | Clinical Meaning |
| :--- | :---: | :--- |
| **`normal`** | `0` | No acute ischemic or hemorrhagic stroke detected |
| **`ischemic`** | `1` | Acute / subacute cerebral infarction / ischemia |
| **`hemorrhagic`** | `2` | Intracranial hemorrhage |

Inverse mapping:
$$\{0: \text{'normal'}, 1: \text{'ischemic'}, 2: \text{'hemorrhagic'}\}$$

---

## 4. Optimization & Loss Formulation

### 4.1 Loss Function
- **Standard Objective**: Categorical Cross-Entropy Loss (`nn.CrossEntropyLoss()`).
- Since our curated training set is balanced (985 normal, 977 ischemic, 985 hemorrhagic), uniform class weighting is standard:
  $$\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(z_{i, y_i})}{\sum_{c=0}^2 \exp(z_{i, c})}$$
- **Label Smoothing**: Optional regularization $\epsilon = 0.05$ to prevent overconfident output distributions on noisy pseudocolor CT borders.

### 4.2 Optimizer & Hyperparameters
- **Optimizer**: AdamW (`weight_decay=1e-2`) or SGD with Nesterov momentum (`momentum=0.9`, `weight_decay=1e-4`).
- **Initial Learning Rate**:
  - Backbone (pretrained layers): $1 \times 10^{-4}$ (or $1 \times 10^{-5}$ if fine-tuning end-to-end).
  - Head (new linear classification layer): $1 \times 10^{-3}$.
- **Batch Size**: 32 (or 16 depending on available hardware memory).

### 4.3 Learning Rate Scheduler
- **Cosine Annealing with Warmup**:
  - Warmup: 3 epochs linear warmup from $\eta_{\text{min}} = 10^{-6}$ to $\eta_{\text{base}}$.
  - Schedule: Cosine decay over remaining epochs to minimum learning rate $10^{-6}$.
- Alternative: `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)` monitoring validation Macro F1.

---

## 5. Regularization, Checkpointing & Early Stopping

### 5.1 Early Stopping Strategy
- **Monitor Metric**: Validation Macro F1 score (or Validation Loss).
- **Patience**: 7–10 epochs.
- If the monitored metric fails to improve for `patience` consecutive epochs, training terminates and the best checkpoint is restored.

### 5.2 Checkpointing Protocol
Checkpoints will be saved in `models/checkpoints/` with systematic naming:
- `ct_baseline_<arch>_best_val_f1.pt`: Model weights yielding the highest validation Macro F1.
- `ct_baseline_<arch>_latest.pt`: Most recent epoch weights (to resume training if interrupted).
- Checkpoint payload must store:
  ```python
  {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'val_f1': val_f1,
      'val_loss': val_loss,
      'class_to_idx': {'normal': 0, 'ischemic': 1, 'hemorrhagic': 2},
      'normalization': {'mean': mean, 'std': std},
      'seed': SEED
  }
  ```

---

## 6. Evaluation Metrics & Clinical Decision Criteria

Given the safety-critical clinical application, aggregate accuracy alone is insufficient. We compute:

1. **Overall Accuracy**: Total correct classifications / total samples.
2. **Per-Class Sensitivity (Recall)**:
   $$\text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
   *Crucial for stroke diagnosis*: False negatives in hemorrhagic stroke (calling it normal or ischemic) can lead to contraindicated thrombolysis (e.g., administering tPA to an active bleeder is fatal).
3. **Per-Class Specificity & Precision**:
   $$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$
4. **Macro-Averaged F1-Score**:
   $$\text{Macro F1} = \frac{1}{3} \sum_{c=0}^2 \text{F1}_c$$
5. **Confusion Matrix**: Full $3 \times 3$ confusion matrix visualization evaluated on the held-out test split ($N = 632$).
6. **Receiver Operating Characteristic (ROC-AUC)**: One-vs-Rest (OvR) multiclass Area Under Curve.

---

## 7. Reproducibility & Experiment Tracking

- **Fixed Random Seed**: `SEED = 42` applied across Python `random`, `numpy`, and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`).
- **Data Splits**: Strictly use pre-generated CSVs (`data/splits/train.csv`, `validation.csv`, `test.csv`).
- **Tracking Log**: Each experiment run will write metadata, hyperparameters, training loss curves, and evaluation metrics into `experiments/logs/ct_baseline_<timestamp>.json`.
