import pandas as pd
import numpy as np

np.random.seed(42)

n = 200

education_levels = [
    "High School",
    "Graduate",
    "Postgraduate",
    "Doctorate"
]

departments = [
    "HR",
    "Finance",
    "Marketing",
    "Sales",
    "IT"
]

states = [
    "Delhi",
    "Maharashtra",
    "Karnataka",
    "Tamil Nadu",
    "Gujarat"
]

data = pd.DataFrame({
    "Customer_ID": [
        f"C{i:04d}" for i in range(1, n + 1)
    ],

    "Age": np.random.randint(18, 65, n),

    "Gender": np.random.choice(
        ["Male", "Female", "Other"],
        n
    ),

    "Education_Level": np.random.choice(
        education_levels,
        n
    ),

    "Income": np.random.randint(
        20000, 150000, n
    ),

    "Satisfaction": np.random.randint(
        1, 6, n
    ),

    "Temperature": np.round(
        np.random.normal(28, 4, n),
        1
    ),

    "Purchase_Date": pd.date_range(
        start="2026-01-01",
        periods=n,
        freq="D"
    ),

    "Department": np.random.choice(
        departments,
        n
    ),

    "Would_Recommend": np.random.choice(
        ["Yes", "No"],
        n
    ),

    "Comments": np.random.choice(
        [
            "Very good service",
            "Good experience",
            "Average experience",
            "Could be improved",
            "Excellent service"
        ],
        n
    ),

    "State": np.random.choice(
        states,
        n
    ),

    "Discount": np.round(
        np.random.uniform(0, 50, n),
        2
    )
})

# Save as Excel
output_file = "data/test_dataset.xlsx"

data.to_excel(
    output_file,
    index=False
)

print(f"Dataset created successfully: {output_file}")
print(f"Rows: {len(data)}")
print(f"Columns: {len(data.columns)}")
