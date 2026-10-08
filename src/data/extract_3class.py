from zipfile import ZipFile
from pathlib import Path

ZIP_PATH = "data/raw/CT Kaggle Dataset.zip"
OUTPUT_DIR = Path("data/processed/teknofest_3class")

# Only use clinically unambiguous classes
CLASS_MAP = {
    "İnme Yok": "normal",
    "İskemi": "ischemic",
    "Kanama": "hemorrhagic",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for class_name in CLASS_MAP.values():
    (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

count = {
    "normal": 0,
    "ischemic": 0,
    "hemorrhagic": 0,
}

with ZipFile(ZIP_PATH, "r") as z:

    for name in z.namelist():

        if not name.lower().endswith(".jpg"):
            continue

        parts = name.split("/")

        # dataset / split / class / filename
        if len(parts) < 4:
            continue

        split = parts[1]
        original_class = parts[2]
        filename = parts[3]

        # Ignore competition folders
        if original_class not in CLASS_MAP:
            continue

        new_class = CLASS_MAP[original_class]

        # Keep original split for now.
        # We will NOT trust it for final experiments.
        destination = (
            OUTPUT_DIR
            / new_class
            / split
        )

        destination.mkdir(parents=True, exist_ok=True)

        output_file = destination / filename

        with z.open(name) as source:
            output_file.write_bytes(source.read())

        count[new_class] += 1

print("\nExtraction complete.")
print("=" * 50)

for class_name, total in count.items():
    print(f"{class_name:12}: {total}")

print(f"\nOutput: {OUTPUT_DIR}")
