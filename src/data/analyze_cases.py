import pandas as pd

MANIFEST = "data/metadata/teknofest_3class_manifest.csv"

df = pd.read_csv(MANIFEST)

print("=" * 70)
print("CASE / IMAGE DISTRIBUTION")
print("=" * 70)

# --------------------------------------------------
# Images per case
# --------------------------------------------------

images_per_case = df.groupby("case_id").size()

print("\nImages per case:")
print(images_per_case.describe())

print("\nCases with 1 image:")
print((images_per_case == 1).sum())

print("Cases with >1 image:")
print((images_per_case > 1).sum())

print("Maximum images for one case:")
print(images_per_case.max())


# --------------------------------------------------
# Top cases
# --------------------------------------------------

print("\nTop 20 cases by number of images:")
print(images_per_case.sort_values(ascending=False).head(20))


# --------------------------------------------------
# Class consistency
# --------------------------------------------------

class_per_case = df.groupby("case_id")["class"].nunique()

print("\nCases appearing in multiple classes:")
print((class_per_case > 1).sum())

if (class_per_case > 1).any():

    print("\nExamples:")
    
    multi_class_cases = class_per_case[class_per_case > 1].index

    for case_id in multi_class_cases[:20]:

        subset = df[df["case_id"] == case_id]

        print(
            f"\nCase {case_id}:"
        )

        print(
            subset[
                ["image_path", "class", "original_split"]
            ].to_string(index=False)
        )


# --------------------------------------------------
# Split overlap
# --------------------------------------------------

print("\nCases appearing across original splits:")

split_per_case = df.groupby("case_id")["original_split"].nunique()

print(
    f"Cases in multiple original splits: "
    f"{(split_per_case > 1).sum()}"
)

print(
    f"Cases appearing in all 3 splits: "
    f"{(split_per_case == 3).sum()}"
)
