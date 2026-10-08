from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# ============================================================
# Configuration
# ============================================================

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MANIFEST_PATH = Path("data/metadata/teknofest_3class_manifest.csv")
OUTPUT_DIR = Path("data/splits")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load manifest
# ============================================================

df = pd.read_csv(MANIFEST_PATH)

print(f"Total images: {len(df)}")
print(f"Total case groups: {df['case_id'].nunique()}")
print()


# ============================================================
# Verify each case belongs to only one clinical class
# ============================================================

case_class_counts = df.groupby("case_id")["class"].nunique()

multi_class_cases = case_class_counts[case_class_counts > 1]

if len(multi_class_cases) > 0:
    print("ERROR: Some case groups contain multiple classes.")
    print(multi_class_cases)
    raise ValueError("A case_id belongs to multiple classes.")

print("✓ Every case group belongs to exactly one class.")
print()


# ============================================================
# Create one row per case group
# ============================================================

groups = (
    df.groupby("case_id")["class"]
    .first()
    .reset_index()
)

print("Class distribution at case level:")
print(groups["class"].value_counts())
print()


# ============================================================
# STEP 1: Split TEST = 15%
# Remaining = 85%
# ============================================================

split_test = StratifiedShuffleSplit(
    n_splits=1,
    test_size=TEST_RATIO,
    random_state=SEED
)

remaining_idx, test_idx = next(
    split_test.split(groups["case_id"], groups["class"])
)

remaining_groups = groups.iloc[remaining_idx].copy()
test_groups = groups.iloc[test_idx].copy()


# ============================================================
# STEP 2: Split VALIDATION from remaining
#
# We want:
# Train = 70% total
# Val   = 15% total
#
# Remaining after test = 85%
# Validation fraction of remaining = 15 / 85
# ============================================================

val_fraction_of_remaining = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

split_val = StratifiedShuffleSplit(
    n_splits=1,
    test_size=val_fraction_of_remaining,
    random_state=SEED
)

train_idx, val_idx = next(
    split_val.split(
        remaining_groups["case_id"],
        remaining_groups["class"]
    )
)

train_groups = remaining_groups.iloc[train_idx].copy()
val_groups = remaining_groups.iloc[val_idx].copy()


# ============================================================
# Get actual images belonging to each case group
# ============================================================

train_case_ids = set(train_groups["case_id"])
val_case_ids = set(val_groups["case_id"])
test_case_ids = set(test_groups["case_id"])

train_df = df[df["case_id"].isin(train_case_ids)].copy()
val_df = df[df["case_id"].isin(val_case_ids)].copy()
test_df = df[df["case_id"].isin(test_case_ids)].copy()


# ============================================================
# Add split column
# ============================================================

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"


# ============================================================
# Safety checks
# ============================================================

train_cases = set(train_df["case_id"])
val_cases = set(val_df["case_id"])
test_cases = set(test_df["case_id"])

assert train_cases.isdisjoint(val_cases)
assert train_cases.isdisjoint(test_cases)
assert val_cases.isdisjoint(test_cases)

assert len(train_df) + len(val_df) + len(test_df) == len(df)

all_split_cases = train_cases | val_cases | test_cases

assert all_split_cases == set(df["case_id"].unique())

print("✓ No case group appears in multiple splits.")
print("✓ Every image belongs to exactly one split.")
print()


# ============================================================
# Save split CSV files
# ============================================================

train_path = OUTPUT_DIR / "train.csv"
val_path = OUTPUT_DIR / "validation.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)


# ============================================================
# Print summary
# ============================================================

print("=" * 60)
print("FINAL SPLIT SUMMARY")
print("=" * 60)

for name, split_df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df),
]:

    print(f"\n{name}")
    print("-" * 40)

    print(f"Images : {len(split_df)}")
    print(f"Cases  : {split_df['case_id'].nunique()}")

    print("\nClass distribution:")
    print(split_df["class"].value_counts().sort_index())

    print("\nClass percentages:")
    print(
        (
            split_df["class"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )


# ============================================================
# Overall summary CSV
# ============================================================

summary_rows = []

for name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:

    for class_name in sorted(df["class"].unique()):

        subset = split_df[split_df["class"] == class_name]

        summary_rows.append({
            "split": name,
            "class": class_name,
            "images": len(subset),
            "cases": subset["case_id"].nunique(),
        })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "split_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print(train_path)
print(val_path)
print(test_path)
print(OUTPUT_DIR / "split_summary.csv")
