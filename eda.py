import pandas as pd
import plotly.express as px

df = pd.read_csv("data/processed/cleaned_data.csv")

# Time series
fig = px.line(df, x="date", y="Appliances")
fig.show()

# Correlation
fig = px.imshow(df.corr(numeric_only=True))
fig.show()

print("EDA complete")