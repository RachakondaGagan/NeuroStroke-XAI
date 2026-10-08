# Dataset Methodology & Split Architecture

## 1. Dataset Origin & Source

The brain CT dataset was obtained from the TEKNOFEST 2021 Brain CT archive (`data/raw/CT Kaggle Dataset.zip`).
The raw archive contains 6,570 JPG images arranged under:
```
dataset/{train,validation,test}/{folder_name}/...
```

The source folders are:
1. `İnme Yok` (No stroke / Normal)
2. `İskemi` (Ischemia)
3. `Kanama` (Hemorrhage)
4. `YarısmaVeriSeti_1_Oturum` (Competition Dataset Session 1)
5. `YarısmaVeriSeti_2_Oturum` (Competition Dataset Session 2)

---

## 2. Conservative Clinical Curation Strategy

### 2.1 Exclusion of Ambiguous Competition Folders
The two competition session folders (`YarısmaVeriSeti_1_Oturum` and `YarısmaVeriSeti_2_Oturum`) contain unlabeled or non-explicitly annotated evaluation sets from the competition challenge.
- **Decision**: These folders are **strictly excluded**.
- **Rationale**: In clinical AI systems, training or evaluating on inferred or unverified pseudo-labels introduces high risk of diagnostic miscalibration. Only verified, clinically explicit categorical folders were retained.

### 2.2 Curated 3-Class Cohort
Retaining solely the three explicit clinical classes produced:
- **Normal** (`İnme Yok`): 1,400 images
- **Ischemic** (`İskemi`): 1,400 images
- **Hemorrhagic** (`Kanama`): 1,400 images
- **Total Curated Dataset**: **4,200 images** (perfectly balanced 1:1:1 prior to grouping).

---

## 3. Data Integrity & Modality Audit

A programmatic audit of all 4,200 images established:
- **Readable images**: 4,200 / 4,200 (100%)
- **Corrupted / unreadable images**: 0 (0%)
- **Image color space**: 100% 3-channel RGB (pseudocolor/rendered CT slices)
- **Image dimensions**:
  - `512 × 512`: 4,059 images (96.64%)
  - Non-standard dimensions: 141 images (3.36%) with variable aspect ratios (e.g., `512 × 759`)
- **Key Clinical / Engineering Insight**: The images are JPEG-compressed color-rendered CT representations. They do not contain raw 16-bit DICOM Hounsfield Units (HU). Consequently:
  - We do **not** claim to perform HU-based tissue attenuation analysis.
  - RGB channels are preserved throughout the pipeline.
  - Normalization and preprocessing respect this visual data representation.

---

## 4. Case/Group Leakage Audit & Limitations

### 4.1 Investigation of the Original Kaggle Split
An audit of the original train/validation/test folders provided in the raw archive revealed severe data leakage across splits:
- Slices belonging to the same CT study/subject shared a common filename prefix (e.g., `10007_1.jpg`, `10007_10.jpg` share the prefix `10007`).
- In the original Kaggle split:
  - **490 groups** were distributed across multiple splits.
  - **40 groups** were present simultaneously in all three splits (train, validation, and test).
- Evaluating models on the original split would yield overly optimistic, artificially inflated performance due to slice-level cross-contamination.

### 4.2 Filename-Derived Case/Group Identifier
To combat this leakage, we parsed each filename by stripping the trailing underscore and slice index (`filename.rsplit("_", 1)[0]`).
- **Limitation Note**: Because the dataset lacks patient-level hospital metadata, DICOM headers, or demographic records, this prefix cannot be definitively verified as a unique patient ID.
- **Accurate Terminology**: We formally refer to this entity as a **"filename-derived case/group identifier"**, rather than an authenticated patient ID.

### 4.3 Group Consistency
Audit results for the 4,200 curated images:
- **2,866 unique filename-derived case groups**.
- **Cross-class consistency**: Exactly 0 case groups contained mixed clinical classes. Every case group was 100% class-pure.

---

## 5. Independent Group-Level Split Architecture

To ensure zero leakage across experimental partitions, an independent group-level stratified split was engineered using a fixed seed (`seed = 42`).

### 5.1 Split Methodology
1. Group all 4,200 records by their `case_id` (2,866 unique groups) with their associated class.
2. Stratify at the group level:
   - **Test Set**: 15% of case groups (430 groups).
   - **Validation Set**: 15% of case groups (430 groups).
   - **Training Set**: 70% of case groups (2,006 groups).
3. Map all image slices corresponding to each assigned `case_id` into their respective split.

### 5.2 Split Statistics

The exact split distributions recorded in `data/splits/`:

| Split | Images | % of Images | Case Groups | Hemorrhagic | Ischemic | Normal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** (`train.csv`) | 2,947 | 70.17% | 2,006 | 985 | 977 | 985 |
| **Validation** (`validation.csv`) | 621 | 14.79% | 430 | 200 | 211 | 210 |
| **Test** (`test.csv`) | 632 | 15.05% | 430 | 215 | 212 | 205 |
| **Total** | **4,200** | **100.0%** | **2,866** | **1,400** | **1,400** | **1,400** |

### 5.3 Zero-Leakage Verification
Cross-split set intersection tests verify:
- $\text{Train Groups} \cap \text{Validation Groups} = \emptyset$ (0 overlap)
- $\text{Train Groups} \cap \text{Test Groups} = \emptyset$ (0 overlap)
- $\text{Validation Groups} \cap \text{Test Groups} = \emptyset$ (0 overlap)

All image files referenced in the CSV split manifests exist on disk with 0 missing files.
