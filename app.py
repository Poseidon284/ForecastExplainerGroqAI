import streamlit as st
import pandas as pd
import pickle
from forecast_utils import make_forecast, plot_forecast, mark_pdf
from genai_utils import explain_forecast, setup, get_llm
import io

# Load Prophet model
with open("cake_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📑 MarketBuddy")

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Forecast Table", "📈 Trend Charts", "🤖 AI Insights","EDA and Model", "֎ Chatbot"])

    with tab1:
        st.subheader("Forecasted Sales")
        st.table(level_fore)

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
        st.subheader("How did we arrive here?")
        st.text("""We’ve put these insights together using the power of AI, your past sales patterns and historical trends, compared with the expected sales for the period you selected. To make things simple, we’ve broken it down to show you the average sales per day each month, giving you a quick view of what a typical day ahead might look like. Think of these insights as your guide to boost your sales for this period.
                """)
        st.divider()
        st.markdown(explanation)
        pdf = mark_pdf(explanation)
        out = io.BytesIO()
        pdf.save_bytes(out)
        # assert out.getbuffer().nbytes > 0
        st.download_button(
            label="📥 Download File",
            data=out,
            file_name="MarketBuddyInsights.pdf",
            mime="application/pdf"
        )

    with tab4:
        link = "https://mybinder.org/v2/gh/Poseidon284/ForecastNtbkHost/5438b8dc9f0c97d7343c8684eb1e2f119f3872ee?urlpath=lab%2Ftree%2FProphetForecasting.ipynb"
        st.subheader("Understand more about how we used your sales data here")
        st.text("Open the below link and click on the View on voila icon - The yellow curve in the task bar")
        st.markdown(f"[View EDA]({link})")

    with tab5:
        link2 = "https://tariffsupp.streamlit.app/"
        st.subheader("Chat with your Data")
        st.text("""Ask your data questions and you shall get answers.\n\n(Try something like "Give me the Sales Details for the US for A,B,C and D")""")
        st.markdown(f"[Chatbot]({link2})")