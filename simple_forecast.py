import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Load your CSV, parsing nothing yet
df = pd.read_csv("sales.csv")

# 2. Rename for Prophet
df = df.rename(columns={
    "Date": "ds",
    "Weekly_Sales": "y"
})

# 3. Normalize separators: change all “-” to “/”
df["ds"] = df["ds"].astype(str).str.replace("-", "/", regex=False)

# 4. Parse dates, assuming day-first (handles both d/m/Y and m/d/Y)
df["ds"] = pd.to_datetime(df["ds"], dayfirst=True, infer_datetime_format=True)

# 5. Sort and reset index
df = df.sort_values("ds").reset_index(drop=True)

# 6. Verify
print(df.info())
print(df.head())

# 7. Quick plot
df.set_index("ds")["y"].plot(
    title="Weekly Sales Over Time",
    ylabel="Weekly_Sales"
)


# ─── Step 4: Fit & Forecast with Prophet ────────────────────────────────



# 4.1 Instantiate the model
model = Prophet(
    daily_seasonality=False,   # we have weekly data
    weekly_seasonality=True,
    yearly_seasonality=True
)

# 4.2 Fit on your prepared df (only ds + y columns)
model.fit(df[["ds", "y"]])

# 4.3 Create a DataFrame for future dates & predict
horizon = 30  # how many weeks ahead to forecast
future = model.make_future_dataframe(periods=horizon, freq="W")  
# freq="W" ensures weekly steps; default is daily
forecast = model.predict(future)

# 4.4 Plot the full forecast (history + future)
import matplotlib.pyplot as plt

fig1 = model.plot(forecast)
plt.title(f"Weekly Sales: Historical + {horizon}-Week Forecast")
plt.tight_layout()
plt.show()

# 4.5 Plot forecast components (trend / yearly / weekly)
fig2 = model.plot_components(forecast)
plt.tight_layout()
plt.show()


# ───────────────────────────────────────────────────────────────────────


# 1. Hold out the last 8 weeks
cutoff = df["ds"].max() - pd.Timedelta(weeks=8)
train  = df[df["ds"] <= cutoff]
test   = df[df["ds"]  > cutoff]

# 2. Train on the pre-cutoff data
m2 = Prophet(weekly_seasonality=True, yearly_seasonality=True)
m2.fit(train[["ds","y"]])

# 3. Forecast those 8 weeks on Fridays
future2 = m2.make_future_dataframe(periods=8, freq="W-FRI")
fc2     = m2.predict(future2)[["ds","yhat"]]

# 4. Aggregate your actual test sales to weekly totals
actual_weekly = (
    test
    .groupby("ds")["y"]
    .sum()
    .reset_index(name="y_actual")
)

# 5. Merge only the weeks that appear in both
eval_df = actual_weekly.merge(fc2, on="ds", how="inner")
print(f"Comparing {len(eval_df)} weeks")

# 6. Compute RMSE
mse  = mean_squared_error(eval_df["y_actual"], eval_df["yhat"])
rmse = np.sqrt(mse)
print(f"8-week hold-out RMSE (total sales per week): {rmse:.2f}")
