import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import io
import holidays
import uuid
import numpy as np

eda_df = pd.read_csv("eda_df.csv")
eda_df['date'] = pd.to_datetime(eda_df['date'])

def days_plot(year):
    if year == "All":
        year = eda_df["date"].dt.year.unique()
    else:
        year = [year]
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    totals = []

    for i,day in enumerate(days):
        total = eda_df.loc[(eda_df['date'].dt.year.isin(year)) & (eda_df['date'].dt.dayofweek == i), 'Sales'].sum() #Monday is 0 and Sun is 6
        totals.append(round(total,2))

    plot_df = pd.DataFrame({
        "Days":days,
        "Sales": totals,
    })

    fig1 = px.bar(plot_df, x='Days', y='Sales', text='Sales',
                title=f'Historical Cake Sales by Day of Week for {", ".join(str(x) for x in year)}', 
                labels={'Days':'Day'},
    )

    fig1.update_layout(yaxis_title='Total Cake Sales', xaxis_title='Days',text_auto=".2f", width=700, height=500)
    return [fig1], plot_df

def months_plot(year):
    if year == "All":
        year = eda_df["date"].dt.year.unique()
    else:
        year = [year]
    totals = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for i in range(1, 13): 
        total = eda_df.loc[(eda_df['date'].dt.year.isin(year)) & (eda_df['date'].dt.month == i), 'Sales'].sum()
        totals.append(round(total, 2))

    plot_df = pd.DataFrame({
        "Month": months,
        "Sales": totals
    })

    fig = px.bar(
        plot_df,
        x='Month',
        y='Sales',
        text='Sales',
        text_auto=".2f",
        title=f'Total Cake Sales by Month for {', '.join(str(x) for x in year)}',
    )

    fig.update_layout(yaxis_title='Total Cake Sales', xaxis_title='Month', width=700, height=500)
    return [fig], plot_df

def holiday_analysis():
    new_df = eda_df.copy()
    fr_holidays = holidays.France(years=[2021,2022])
    holiday_dates = pd.to_datetime(list(fr_holidays.keys()))
    new_df['is_holiday_or_before'] = eda_df['date'].isin(holiday_dates) | eda_df['date'].isin(
        [holiday_dates - pd.Timedelta(days=1)]
    )

    new_df["Quarter"] = new_df['date'].dt.to_period("Q").astype(str)
    grouped = (
        new_df.groupby(['Quarter','is_holiday_or_before'])['Sales']
        .mean()
        .reset_index()
    )
    grouped["Day Type"] = grouped["is_holiday_or_before"].map(
        {True: "Holiday/Before", False: "Non-Holiday"}
    )
    fig0 = px.bar(
        grouped,
        x="Quarter",
        y="Sales",
        color="Day Type",
        barmode="group",
        text_auto=".2f",
        title="Quarterly Average Sales: Holidays vs Non-Holidays",
        labels={"Quarter": "Quarter", "Sales": "Average Sales"}
    )

    fig0.update_layout(
        xaxis=dict(title="Quarter"),
        yaxis=dict(title="Average Sales")
    )

    non_holiday_sales = new_df.loc[~new_df['is_holiday_or_before'], 'Sales'].mean()
    holiday_mean_sales = new_df.loc[new_df["is_holiday_or_before"], 'Sales'].mean()

    fig1 = px.bar(
        x=['Non-Holiday Sales','Holiday Sales'], 
        y=[non_holiday_sales, holiday_mean_sales], 
        title="Holiday Effect on Sales(Holiday and one day before vs Other days)",
        text_auto=".2f",
    )
    fig1.update_layout(
        xaxis=dict(title="Holidays"),
        yaxis=dict(title="Average Sales"),
    )

    fig = [fig0, fig1]
    return fig, grouped

def sales_bar(period = 'M', year = "All"):
    if year == "All":
        year = eda_df["date"].dt.year.unique()
    else:
        year = [year]
    if period == 'M':
        eda_df["Period"] = eda_df["date"].dt.to_period("M")
        periodical_sales = (
            eda_df[eda_df["date"].dt.year.isin(year)].groupby("Period")["Sales"]
            .mean()
            .reset_index()
        )
    elif period == 'Q':
        eda_df["Period"] = eda_df["date"].dt.to_period("Q")
        periodical_sales = (
            eda_df[eda_df["date"].dt.year.isin(year)].groupby("Period")["Sales"]
            .mean()
            .reset_index()
        )
    periodical_sales["Period"] = periodical_sales["Period"].astype(str)
    fig = px.bar(
        periodical_sales,
        x="Period",
        y="Sales",
        labels={"Period": "Months", "Sales": "Sales"},
    )
    fig.update_layout(bargap=0.15,xaxis_tickangle=-45, yaxis=dict(tickprefix='€'))
    if period == 'M':
        fig.update_layout(title=f"Average Daily Sales per Month for {", ".join(str(x) for x in year)}")
    else:
        fig.update_layout(title=f"Average Daily Sales per Quarter for {", ".join(str(x) for x in year)}")
    return [fig], periodical_sales

def evaluate_forecast(forecast):    
    width = forecast['yhat_upper'] - forecast['yhat_lower']
    
    width_pct = width / forecast['yhat'].replace(0, np.nan) * 100
    
    confidence_score = 100 - width_pct
    confidence_score = confidence_score.fillna(0).mean()
    
    interval_width_pct = width_pct.fillna(0).mean()
    
    return confidence_score, interval_width_pct


def download_plotly_chart(fig, filename="chart", format="html"):
    random_key = str(uuid.uuid4())
    # if format.lower() == "png":
    #     img_bytes = fig.to_image(format="png")
    #     st.download_button(
    #         label=f"📥 Download Chart",
    #         data=img_bytes,
    #         file_name=f"{filename}.png",
    #         key=f"0{random_key}",
    #         mime="image/png"
    #     )
    if format.lower() == "html":
        html_str = fig.to_html(full_html=True)
        st.download_button(
            label=f"📥 Download Chart",
            data=html_str,
            file_name=f"{filename}.html",
            key=f"1{random_key}",
            mime="text/html"
        )
    else:
        st.error("Unsupported format. Use 'png' or 'html'.")

def trend_charts(level_fore, periods, fperiod='P'):
    if fperiod == 'P':
        plot_df = level_fore[:-periods].copy()
    elif fperiod == 'F':
        plot_df = level_fore[-periods:].copy()
    plot_df["Period"] = range(1, len(plot_df) + 1)
    if fperiod == 'P':
        fig_line = px.line(plot_df, x='Period', y='Average Sales', markers=True)
    elif fperiod == 'F':
        fig_line = px.bar(plot_df, x='Period', y='Average Sales')
    fig_line.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=plot_df["Period"],
                ticktext=plot_df["Date"].astype(str),
                showticklabels=False, 
                title="Periods"
            )
        )
    
    return [fig_line]