from pathlib import Path
import pandas as pd

DATASET_DIR = Path("data/processed/teknofest_3class")
OUTPUT_PATH = Path("data/metadata/teknofest_3class_manifest.csv")

records = []

for class_name in ["normal", "ischemic", "hemorrhagic"]:

    class_dir = DATASET_DIR / class_name

    for image_path in class_dir.rglob("*.jpg"):

        # Expected structure:
        # class / split / filename.jpg
        relative = image_path.relative_to(class_dir)
        parts = relative.parts

        if len(parts) < 2:
            continue

        original_split = parts[0]
        filename = image_path.name

        # Example:
        # 10113_8.jpg -> 10113
        stem = image_path.stem

        if "_" in stem:
            case_id = stem.rsplit("_", 1)[0]
        else:
            case_id = stem

        records.append({
            "image_path": str(image_path),
            "class": class_name,
            "original_split": original_split,
            "case_id": case_id,
        })

df = pd.DataFrame(records)

df = df.sort_values(
    ["class", "original_split", "case_id", "image_path"]
).reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("TEKNOFEST MANIFEST CREATED")
print("=" * 70)

print(f"\nTotal records: {len(df)}")
print(f"Output: {OUTPUT_PATH}")

print("\nClass distribution:")
print(df["class"].value_counts())

print("\nOriginal split distribution:")
print(df["original_split"].value_counts())

print("\nUnique case prefixes:")
print(df["case_id"].nunique())

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))
