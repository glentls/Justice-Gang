import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\\glen\\Desktop\\tiktokhackathon\\sampledata.csv")

print(df.head())
pd.set_option('display.max_columns', None)
print(df.columns)

df = df[['feature_id', 'feature_name', 'feature_description']]
df['feature_id'] = range(1, len(df) + 1)

stratify_col = None

train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=stratify_col
)

val_size = 0.15 / (1 - 0.15)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_size,
    random_state=42,
    stratify=None
)

print("Train:", len(train_df), "Validation:", len(val_df), "Test:", len(test_df))
print(train_df.columns)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Saved datasets: Train({len(train_df)}), Validation({len(val_df)}), Test({len(test_df)})")