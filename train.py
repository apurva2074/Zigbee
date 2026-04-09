import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("data/processed/cleaned_data.csv")

# Split
train_size = int(len(df) * 0.8)

train = df[:train_size]
test = df[train_size:]

X_train = train.drop(['Appliances', 'date'], axis=1)
y_train = train['Appliances']

X_test = test.drop(['Appliances', 'date'], axis=1)
y_test = test['Appliances']

# Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, preds)
print(f"MAE: {mae:.2f}")

# Save model
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved!")