import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
from markdown_pdf import MarkdownPdf, Section

def make_forecast(model, freq, periods: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

import pandas as pd
import plotly.graph_objects as go

def prophet_line_plot(model, forecast):
    # Extract actuals from the model
    history = model.history.copy()
    history["ds"] = pd.to_datetime(history["ds"])
    
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    cutoff = history["ds"].max()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history["ds"], y=history["y"],
        mode="lines",
        name="Actual",
        line=dict(color="blue", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color="red", width=2)
    ))

    fig.add_vrect(
        x0=cutoff,
        x1=forecast["ds"].max(),
        fillcolor="red",
        opacity=0.1,
        line_width=0,
        annotation_text="Forecast Period",
        annotation_position="top left"
    )

    fig.update_layout(
        title="Forecasted Sales",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white"
    )

    return fig


def plot_forecast(model, forecast):
    fig = prophet_line_plot(model, forecast)
    fig2 = plot_components_plotly(model, forecast)
    return [fig,fig2]

def mark_pdf(explanation):
    pdf = MarkdownPdf(toc_level=2, optimize=True)
    pdf.add_section(Section(explanation))
    return pdf
