import streamlit as st
import pandas as pd
import pickle
from forecast_utils import make_forecast, plot_forecast
from genai_utils import explain_forecast, setup, get_llm

# Load Prophet model
with open("cake_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🤖 AI Forecasting and Insights App")

option = st.selectbox(
    "Select Forecast Horizon:",
    ("Weeks", "Months", "Quarters")
)

# Input horizon value
if option == "Weeks":
    periods = st.number_input("Enter number of weeks:", min_value=1, max_value=4, value=4)
    horizon_days = periods * 7
    agg_level = "D"   # daily forecast
elif option == "Months":
    periods = st.number_input("Enter number of months:", min_value=1, max_value=12, value=4)
    horizon_days = periods * 30
    agg_level = "M"   # monthly aggregation
elif option == "Quarters":
    periods = st.number_input("Enter number of quarters:", min_value=1, max_value=4, value=1)
    horizon_days = periods * 90
    agg_level = "Q"   # quarterly aggregation

# Forecast button
if st.button("Generate Forecast"):
    # Always forecast daily
    forecast = make_forecast(model, "D", horizon_days)

    # Aggregation logic
    if agg_level == "D":
        disp_fore = forecast[["ds", "trend", "yhat"]].copy()
    else:
        forecast["Period"] = pd.to_datetime(forecast["ds"])
        forecast.set_index("Period", inplace=True)
        disp_fore = forecast.resample(agg_level).mean(numeric_only=True).reset_index()
        disp_fore.rename(columns={"Period": "Date"}, inplace=True)

    # Rename cols
    disp_fore = disp_fore.rename(columns={"yhat": "Average Sales", "ds": "Date"})
    disp_fore.index = disp_fore.index + 1

    # Custom aggregation
    if agg_level == "M":
        level_fore = (
            disp_fore[["Date", "Average Sales"]]
            .groupby(disp_fore["Date"].dt.to_period("M"))
            .mean()
        )
    elif agg_level == "Q":
        level_fore = (
            disp_fore[["Date", "Average Sales"]]
            .groupby(disp_fore["Date"].dt.to_period("Q"))
            .mean()
        )
    else:
        level_fore = (
            disp_fore[["Date", "Average Sales"]]
            .groupby(disp_fore["Date"].dt.to_period("W"))
            .mean()
        )
    level_fore = level_fore.drop("Date",axis=1)
    level_fore = level_fore.reset_index()
    api_key = setup("GROQ_API_KEY")
    llm = get_llm(api_key)
    explanation = explain_forecast(level_fore, llm, periods, option)
    # Tabs for UI
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Table", "📈 Trend Charts", "🤖 AI Insights"])

    with tab1:
        st.subheader("Forecasted Sales")
        st.table(level_fore)

        # Download option
        st.download_button(
            "Download Forecast CSV",
            data=forecast.to_csv(index=False),
            file_name="forecast.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("Trend Chart")
        figs = plot_forecast(model, forecast)
        st.plotly_chart(figs[0], use_container_width=True)
        st.subheader("Seasonality Charts")
        st.plotly_chart(figs[1], use_container_width=True)

    with tab3:
        st.subheader("Forecast Explanation")
        st.markdown(explanation)
