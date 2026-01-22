import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/diabetes_cleaned.csv")
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

test_size = 0.3
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    stratify=y,
    random_state=random_state,
)

print("Saving train/test splits to data/ folder...")

# Reconstruct dataframes to save them easily
train_df = X_train.copy()
train_df['Outcome'] = y_train

test_df = X_test.copy()
test_df['Outcome'] = y_test

train_df.to_csv("data/train_split.csv", index=False)
test_df.to_csv("data/test_split.csv", index=False)