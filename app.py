import streamlit as st
import pandas as pd
from prophet import Prophet

@st.cache_data
def load_data():
    df = pd.read_csv("sales.csv")
    df = df.rename(columns={"Date":"ds","Weekly_Sales":"y"})
    df["ds"] = (
        df["ds"]
        .astype(str)
        .str.replace("-", "/", regex=False)
        .pipe(pd.to_datetime, dayfirst=True)
    )
    return df.sort_values("ds").reset_index(drop=True)

@st.cache_resource
def train_model(df):
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df[["ds","y"]])
    return m

st.title("Weekly Sales Forecasting")

df    = load_data()
model = train_model(df)

weeks   = st.slider("Weeks to forecast", 4, 52, 12)
future  = model.make_future_dataframe(periods=weeks, freq="W-FRI")
forecast = model.predict(future)

# Combine actual vs forecast
plot_df = pd.DataFrame({
    "Actual":   df.groupby("ds")["y"].sum(),
    "Forecast": forecast.set_index("ds")["yhat"]
})

st.line_chart(plot_df)

if st.checkbox("Show forecast components"):
    fig = model.plot_components(forecast)
    st.pyplot(fig)
