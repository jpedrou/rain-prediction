import pandas as pd

# Load Data
df = pd.read_csv("../../data/raw/weatherAUS.csv")
df["RainToday"] = df["RainToday"].fillna(df["RainToday"].mode()[0])
df["RainTomorrow"] = df["RainTomorrow"].fillna(df["RainTomorrow"].mode()[0])

df.drop(["Location"], axis=1, inplace=True)

df.shape

df = df.sample(15000, random_state=0)
df.reset_index(drop=True, inplace=True)

# Separate into Train and Test sets
test_df = df.sample(1000, random_state=0)

train_df = df.drop(test_df.index, axis=0)
train_df.reset_index(drop=True, inplace=True)

test_df.reset_index(drop=True, inplace=True)

# Export sets
train_df.to_csv("../../data/processed/df_processed.csv", index=None)
test_df.to_csv("../../data/processed/df_test_processed.csv", index=None)
