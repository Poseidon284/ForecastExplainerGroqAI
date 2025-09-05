import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
from markdown_pdf import MarkdownPdf, Section

def make_forecast(model, freq, periods: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def plot_forecast(model, forecast):
    fig = plot_plotly(model, forecast)
    fig2 = plot_components_plotly(model, forecast)
    return [fig,fig2]

def mark_pdf(explanation):
    pdf = MarkdownPdf(toc_level=2, optimize=True)
    pdf.add_section(Section(explanation))
    return pdf
