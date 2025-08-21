import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def setup(key: str):
    if "STREAMLIT_RUNTIME" in os.environ:
        try:
            return st.secrets[key]
        except:
            st.error("GROQ_API_KEY not found in api.env")
            st.stop()
    else:
        try:
            load_dotenv("api.env")
            groq_api_key = os.getenv(key)
            return groq_api_key
        except:
            st.error("GROQ_API_KEY not found in api.env")
            st.stop()
            
def get_llm(api_key):
    return ChatGroq(
        api_key=api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.7
    )

# def aggre(forecast):
#     # forecast['ds'] = pd.to_datetime(forecast['ds'])
#     if len(forecast) >= 30:
#         forecast = (forecast.groupby(pd.Grouper(key="ds", freq="ME")).mean().reset_index())
#     else:
#         forecast = (forecast.groupby(pd.Grouper(key="ds", freq="W")).mean().reset_index())
#     return forecast

def explain_forecast(forecast, llm, period, freq):
    context_csv = forecast[["ds", "trend" , "yhat", "yhat_upper", "yhat_lower"]]
    forecast_data = context_csv[:-period]
    # forecast = aggre(forecast_data)
    history = pd.read_csv('monthly_data_cakes.csv', index_col=0)
    forecast = forecast_data.to_string()
    dates = history['date'].tolist()
    sales = history['Sales'].tolist()

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert data story-teller tasked with explaining the forecast to a manager who wants to see how well his company is performing and what to look forward to based on this data"
        Historical Sales Data:\n{dates,sales}. This data is from France so if there is a French holiday with high average, tell that.
        These terms are banned - yhat, trend, forecast. There is no geographic focus on the data.
        Any outliers could be explained by holidays in France or popular tourist times."""),
        ("user",f"""Forecasted Sales Data:\n{forecast_data, freq}. This forecast is the average for the given frequency.
         If it is "M" then it is average forecast is for that month, If it is "W" then it is average forecast for the week. If it is "QE" then it is average forecast for next quarter.
         Give some actionable insights in 8 bullet points that will be within 15-25 lines. 
         DO NOT GIVE SLIDE SEPARATIONS "Slide 3 - What to do with this information" 
         Give a concluding statement on what you think will happen to the Sales for the forecasted period.
         Do not use any complicated statistical terms.
         Give output in a markdown friendly format""")
    ])

    chain = prompt_template | llm
    response = chain.invoke({})
    return response.content
