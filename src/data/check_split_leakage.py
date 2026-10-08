from zipfile import ZipFile
from collections import defaultdict, Counter

ZIP_PATH = "data/raw/CT Kaggle Dataset.zip"

# Store:
# split -> patient/case prefix -> list of (class, filename)
cases = defaultdict(lambda: defaultdict(list))

with ZipFile(ZIP_PATH, "r") as z:
    for name in z.namelist():

        if not name.lower().endswith(".jpg"):
            continue

        parts = name.split("/")

        # Expected:
        # dataset / split / class / image.jpg
        if len(parts) < 4:
            continue

        split = parts[1]
        class_name = parts[2]
        filename = parts[-1]

        # Example:
        # 10007_10.jpg -> 10007
        case_id = filename.rsplit(".", 1)[0].rsplit("_", 1)[0]

        cases[case_id][split].append((class_name, filename))


# --------------------------------------------------
# 1. Number of unique case IDs
# --------------------------------------------------

print("=" * 80)
print("CASE / PATIENT PREFIX AUDIT")
print("=" * 80)

print(f"\nUnique case prefixes: {len(cases)}")


# --------------------------------------------------
# 2. Cases appearing in multiple splits
# --------------------------------------------------

leakage_cases = []

for case_id, split_data in cases.items():

    splits = set(split_data.keys())

    if len(splits) > 1:
        leakage_cases.append((case_id, split_data))


print(f"Cases appearing in multiple splits: {len(leakage_cases)}")


# Show first 30
print("\nExamples:")
print("-" * 80)

for case_id, split_data in leakage_cases[:30]:

    print(f"\nCase: {case_id}")

    for split, images in sorted(split_data.items()):

        classes = Counter(class_name for class_name, _ in images)

        print(
            f"  {split:12} | "
            f"{len(images):3} images | "
            f"classes: {dict(classes)}"
        )


# --------------------------------------------------
# 3. Check whether a case appears with multiple labels
# --------------------------------------------------

multi_class_cases = []

for case_id, split_data in cases.items():

    classes = set()

    for images in split_data.values():
        for class_name, _ in images:
            classes.add(class_name)

    if len(classes) > 1:
        multi_class_cases.append((case_id, classes))


print("\n")
print("=" * 80)
print("CLASS CONSISTENCY AUDIT")
print("=" * 80)

print(
    f"\nCases appearing with multiple folder labels: "
    f"{len(multi_class_cases)}"
)

for case_id, classes in multi_class_cases[:30]:

    print(f"Case {case_id}: {sorted(classes)}")


# --------------------------------------------------
# 4. Case counts by split
# --------------------------------------------------

print("\n")
print("=" * 80)
print("UNIQUE CASES BY SPLIT")
print("=" * 80)

split_cases = defaultdict(set)

for case_id, split_data in cases.items():

    for split in split_data:
        split_cases[split].add(case_id)

for split in sorted(split_cases):

    print(
        f"{split:12} : "
        f"{len(split_cases[split])} unique cases"
    )
