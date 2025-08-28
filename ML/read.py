import pandas as pd
# Replace with your actual file path
df = pd.read_csv("C:\\Users\\jacel\\Desktop\\tiktokhackathon\\sampledata.csv")

# Show the first 5 rows
print(df.head())

# Show column names
pd.set_option('display.max_columns', None)
print(df.columns)

# Combine title + description into one column
df['input_text'] = "Feature Name: " + df['feature_name'] + ". Description: " + df['feature_description']

# Define desired column order
desired_order = [
    "feature_id",
    "feature_name",
    "feature_description",
    "flag",
    "reasoning",
    "related_regulations",  
    "input_text"
]

# Reorder dataframe
df = df[desired_order]

# split dataset
from sklearn.model_selection import train_test_split

# If flag exists, you can stratify; if not, just split randomly
stratify_col = df['flag'] if 'flag' in df.columns and df['flag'].notna().any() else None

# --- Step 4a: Split off test set (~15%) ---
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=stratify_col
)

# --- Step 4b: Split train/validation (~70% train, ~15% validation) ---
# Calculate relative validation size
val_size = 0.15 / (1 - 0.15)  # 0.15 / 0.85 ≈ 0.176

train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_size,
    random_state=42,
    stratify=train_val_df['flag'] if stratify_col is not None else None
)

# --- Step 4c: Check sizes ---
print("Train:", len(train_df), "Validation:", len(val_df), "Test:", len(test_df))
print(train_df.columns)

# --- Step 6: Save CSVs ---
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Optional: print confirmation
print(f"Saved datasets: Train({len(train_df)}), Validation({len(val_df)}), Test({len(test_df)})")

