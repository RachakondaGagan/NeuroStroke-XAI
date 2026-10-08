from pathlib import Path
from collections import Counter
from PIL import Image

DATASET_DIR = Path("data/processed/teknofest_3class")

classes = ["normal", "ischemic", "hemorrhagic"]

total = 0
corrupted = []
dimensions = Counter()
modes = Counter()
class_counts = Counter()

for class_name in classes:

    class_dir = DATASET_DIR / class_name

    for image_path in class_dir.rglob("*.jpg"):

        total += 1
        class_counts[class_name] += 1

        try:
            with Image.open(image_path) as img:

                # Verify that the image can actually be decoded
                img.verify()

            # Reopen because verify() closes/invalidates the image object
            with Image.open(image_path) as img:
                dimensions[img.size] += 1
                modes[img.mode] += 1

        except Exception as e:
            corrupted.append((str(image_path), str(e)))


print("=" * 70)
print("TEKNOFEST 3-CLASS IMAGE AUDIT")
print("=" * 70)

print(f"\nTotal images: {total}")

print("\nClass distribution:")
for class_name in classes:
    print(f"  {class_name:12}: {class_counts[class_name]}")

print("\nImage dimensions:")
for dimension, count in dimensions.most_common():
    print(f"  {dimension}: {count}")

print("\nImage modes:")
for mode, count in modes.most_common():
    print(f"  {mode}: {count}")

print(f"\nCorrupted/unreadable images: {len(corrupted)}")

if corrupted:
    print("\nFirst 10 corrupted files:")
    for path, error in corrupted[:10]:
        print(f"  {path}")
        print(f"    {error}")

print("\nAudit complete.")
