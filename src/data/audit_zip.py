from zipfile import ZipFile
from collections import Counter, defaultdict

ZIP_PATH = "data/raw/CT Kaggle Dataset.zip"

with ZipFile(ZIP_PATH, "r") as z:
    files = [
        name for name in z.namelist()
        if name.lower().endswith(".jpg")
    ]

folder_counts = Counter()
folder_samples = defaultdict(list)

for name in files:
    parts = name.split("/")

    # dataset / split / class / image.jpg
    if len(parts) >= 4:
        split = parts[1]
        class_name = parts[2]
        filename = parts[-1]

        folder_counts[(split, class_name)] += 1

        if len(folder_samples[(split, class_name)]) < 3:
            folder_samples[(split, class_name)].append(filename)

print(f"Total JPG images: {len(files)}\n")

print("DATASET STRUCTURE")
print("=" * 80)

for (split, class_name), count in sorted(folder_counts.items()):
    print(f"\nSplit : {split}")
    print(f"Class : {class_name}")
    print(f"Count : {count}")
    print("Sample:", ", ".join(folder_samples[(split, class_name)]))