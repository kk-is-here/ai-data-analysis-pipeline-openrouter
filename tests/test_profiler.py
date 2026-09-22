import json

EXPECTED = {
    "Customer_ID": ("identifier", "none", "not_applicable"),
    "Age": ("numeric", "none", "ratio"),
    "Gender": ("categorical", "none", "nominal"),
    "Education_Level": ("categorical", "none", "ordinal"),
    "Income": ("numeric", "currency", "ratio"),
    "Satisfaction": ("categorical", "none", "ordinal"),
    "Temperature": ("numeric", "none", "interval"),
    "Purchase_Date": ("datetime", "none", "not_applicable"),
    "Department": ("categorical", "none", "nominal"),
    "Would_Recommend": ("binary", "none", "nominal"),
    "Comments": ("text", "none", "not_applicable"),
    "State": ("geographic", "none", "not_applicable"),
    "Discount": ("numeric", "percentage", "ratio"),
}

with open("output/data_dictionary.json", "r", encoding="utf-8") as file:
    results = json.load(file)

correct = 0
total = len(EXPECTED)

for item in results:
    column = item["column_name"]

    actual = (
        item["semantic_type"],
        item.get("semantic_subtype", "none"),
        item.get("measurement_scale")
    )

    expected = EXPECTED.get(column)

    if actual == expected:
        print(f"PASS: {column}")
        correct += 1
    else:
        print(f"FAIL: {column}")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")

print()
print(f"Accuracy: {correct}/{total} ({correct / total * 100:.1f}%)")