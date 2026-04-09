import pandas as pd

print("🔄 Starting preprocessing...")

df = pd.read_csv("data/raw/energydata_complete.csv")

# Convert date
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Drop useless columns
df = df.drop(['rv1', 'rv2'], axis=1)

# Time features
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = df['day'].apply(lambda x: 1 if x >= 5 else 0)

# Lag features
df['lag_1'] = df['Appliances'].shift(1)
df['lag_3'] = df['Appliances'].shift(3)
df['lag_6'] = df['Appliances'].shift(6)

df = df.dropna()

# Save
df.to_csv("data/processed/cleaned_data.csv", index=False)

print("✅ Preprocessing done!")