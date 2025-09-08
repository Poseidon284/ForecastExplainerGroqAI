import streamlit as st
import pandas as pd
import pickle
from forecast_utils import make_forecast, plot_forecast, mark_pdf
from genai_utils import explain_forecast, setup, get_llm, explain_performance
from EDA_utils import trend_charts, download_plotly_chart, evaluate_forecast, sales_bar, days_plot, months_plot
import io

# Load Prophet model
with open("cake_model.pkl", "rb") as f:
    model = pickle.load(f)

api_key = setup("GROQ_API_KEY")
llm = get_llm(api_key)

st.title("📑 MarketBuddy")

option = st.selectbox(
    "Select Forecast Horizon:",
    ("Weeks", "Months", "Quarters")
)

# Input horizon value
if option == "Weeks":
    periods = st.number_input("Enter number of weeks:", min_value=1, max_value=4, value=1)
    horizon_days = periods * 7
    agg_level = "D"
elif option == "Months":
    periods = st.number_input("Enter number of months:", min_value=1, max_value=6, value=1)
    horizon_days = periods * 30
    agg_level = "M"
elif option == "Quarters":
    periods = st.number_input("Enter number of quarters:", min_value=1, max_value=2, value=1)
    horizon_days = periods * 90
    agg_level = "Q"

# Forecast button
if st.button("Generate Forecast"):
    # Always forecast daily
    forecast = make_forecast(model, "D", horizon_days)
    conf_score, interval_pct = evaluate_forecast(forecast)
    figs = plot_forecast(model, forecast)

    # Save in session_state so they persist
    st.session_state["forecast"] = forecast
    st.session_state["figs"] = figs

    # Reset chart choice when new forecast is generated
    st.session_state["chart_choice"] = "Trend Chart"
    st.session_state["sales_choice"] = "Forecasts"

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
    level_fore = level_fore.drop("Date", axis=1)
    level_fore = level_fore.reset_index()

    st.session_state["level_fore"] = level_fore
    st.session_state["periods"] = periods
    st.session_state["agg_option"] = option

    # AI explanation
    st.session_state["explanation"] = explain_forecast(level_fore, llm, periods, option)

# --------------------
# Tabs (always exist)
# --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Forecast Table", "📈 Trend Charts", "🤖 AI Insights", "EDA", "֎ Chatbot"]
)

with tab1:
    if "level_fore" in st.session_state:
        level_fore = st.session_state["level_fore"]
        forecast = st.session_state["forecast"]
        periods = st.session_state["periods"]

        sales_choice = st.selectbox(
            "Select chart to display:",
            ("Past Sales", "Forecasts"),
            index=0,
            key="sales_choice"
        )

        if st.session_state.sales_choice == "Past Sales":
            st.subheader("Past Sales")
            fig_line = trend_charts(level_fore, periods, 'P')
            st.plotly_chart(fig_line, use_container_width=True)
            st.text("This is your average sales for the past five periods you selected.")
            st.table(level_fore[:-periods].tail().reset_index(drop=True))
            dl_option = st.selectbox("Select Format to download", ("png","html"), key="sales")
            download_plotly_chart(fig_line, filename="past_sales_chart", format=dl_option)
            st.info("Your file is ready to be downloaded!")

        if st.session_state.sales_choice == "Forecasts":
            st.subheader("Forecast Sales")
            fig_line2 = trend_charts(level_fore, periods, 'F')
            fig_line2.update_layout(
                yaxis=dict(range=[0,1500])
            )
            st.plotly_chart(fig_line2, use_container_width=True)
            st.text("This is your forecasted average sales for the period you selected.")
            st.write(f"Total Estimated Revenue in this Forecast period : € **{forecast['yhat'][-horizon_days:].sum().round(2)}**")
            st.table(level_fore[-periods:].reset_index(drop=True))
            dl_option = st.selectbox("Select Format to download", ("png","html"), key="sales")
            download_plotly_chart(fig_line2, filename="forecast_sales_chart", format=dl_option)
            st.info("Your file is ready to be downloaded!")
        st.download_button(
            "Download Forecast CSV",
            data=forecast.to_csv(index=False),
            file_name="forecast.csv",
            mime="text/csv"
        )
    else:
        st.info("Click **Generate Forecast** to see the forecast table.")

with tab2:
    if "figs" in st.session_state:
        figs = st.session_state["figs"]

        chart_choice = st.selectbox(
            "Select chart to display:",
            ("Trend Chart", "Seasonality Chart"),
            index=0,
            key="chart_choice"
        )

        if st.session_state.chart_choice == "Trend Chart":
            st.subheader("Trend Chart")
            st.plotly_chart(figs[0], use_container_width=True)
            dl_option = st.selectbox("Select Format to download", ("html"), key="trends")
            download_plotly_chart(figs[0], filename="past_sales_chart", format=dl_option)
            st.info("Your file is ready to be downloaded!")

        elif st.session_state.chart_choice == "Seasonality Chart":
            st.subheader("Seasonality Chart")
            st.plotly_chart(figs[1], use_container_width=True)
            dl_option = st.selectbox("Select Format to download", ("html"), key="trends")
            download_plotly_chart(figs[1], filename="seasonality_chart", format=dl_option)
            st.info("Your file is ready to be downloaded!")
    else:
        st.info("Click **Generate Forecast** to see charts.")

with tab3:
    if "explanation" in st.session_state:
        explanation = st.session_state["explanation"]

        st.subheader("How did we arrive here?")
        st.text(
            """We’ve put these insights together using the power of AI,\
 your past sales patterns and historical trends, compared with the expected sales for the period you selected. To\
 make things simple, we’ve broken it down to show you the average sales per day each month, giving you a quick view\
 of what a typical day ahead might look like. Think of these insights as your guide to boost your sales for this period.
""")
        st.divider()
        st.markdown(explanation)

        pdf = mark_pdf(explanation)
        out = io.BytesIO()
        pdf.save_bytes(out)

        st.download_button(
            label="📥 Download File",
            data=out,
            file_name="MarketBuddyInsights.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Click **Generate Forecast** to see AI insights.")

with tab4:
    # link = "https://mybinder.org/v2/gh/Poseidon284/ForecastNtbkHost/5438b8dc9f0c97d7343c8684eb1e2f119f3872ee?urlpath=lab%2Ftree%2FProphetForecasting.ipynb"
    st.subheader("Understand more about your sales data here")
    # st.text("Open the below link and click on the View on voila icon - The yellow curve in the task bar")
    # st.markdown(f"[View EDA]({link})")
    st.subheader("View your data")
    plot_choice = st.selectbox("View sales by:", ["Monthly Average", "Quarterly Average", "Months" ,"Days"], index=0)
    if plot_choice == 'Monthly Average':
        fig, periodical_sales = sales_bar('M')
    elif plot_choice == 'Quarterly Average':
        fig, periodical_sales = sales_bar('Q')
    elif plot_choice == 'Months':
        fig, periodical_sales = months_plot()
    elif plot_choice == "Days":
        fig, periodical_sales = days_plot()
    st.plotly_chart(fig)
    sales_explanation = explain_performance(periodical_sales, llm, plot_choice)
    st.markdown(sales_explanation)
    
with tab5:
    link2 = "https://tariffsupp.streamlit.app/"
    st.title("Chat with your Data")
    st.text(
        """Ask your data questions and you shall get answers.

        (Try something like "Give me the Sales Details for the US for A,B,C and D")"""
    )
    st.markdown(f"[Chatbot]({link2})")
