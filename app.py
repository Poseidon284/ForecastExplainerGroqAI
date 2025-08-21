import streamlit as st
import pandas as pd
import pickle
from forecast_utils import make_forecast, plot_forecast
from genai_utils import explain_forecast, setup, get_llm

# Load Prophet model
with open("cake_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📈 Prophet Forecasting App with Groq LLM")

option = st.selectbox(
    "Select Forecast Horizon:",
    ("Weeks", "Months", "Quarters")
)

# Input horizon value
if option == "Weeks":
    periods = st.number_input("Enter number of weeks:", min_value=1, max_value=4, value=4)
    freq = "W"
elif option == "Months":
    periods = st.number_input("Enter number of months:", min_value=1, max_value=12, value=4)
    freq = "ME"
elif option == "Quarters":
    periods = st.number_input("Enter number of quarters:", min_value=1, max_value=4, value=1)
    freq = "QE"

# Forecast button
if st.button("Generate Forecast"):
    forecast = make_forecast(model, freq, periods)

    # Show plot
    fig = plot_forecast(model, forecast)
    st.plotly_chart(fig[0], use_container_width=True)
    st.plotly_chart(fig[1], use_container_width=True)

    # Save forecast to CSV
    forecast.to_csv("forecast.csv", index=False)
    st.download_button("Download Forecast CSV", data=forecast.to_csv(index=False), file_name="forecast.csv", mime="text/csv")

    # GenAI explanation
    st.subheader("🤖 Forecast Explanation")
    api_key = setup("GROQ_API_KEY")
    llm = get_llm(api_key)
    explanation = explain_forecast(forecast, llm, periods, freq)
    st.markdown(explanation)
