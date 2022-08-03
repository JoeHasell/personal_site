import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, render, ui
from pathlib import Path


infile = Path(__file__).parent / 'data.csv'
df = pd.read_csv(infile)

app_ui = ui.page_fluid(
    ui.h2("Hello Shiny!"),
    ui.input_slider("max_a", "Slider input", min=0, max=10, value=5),
    ui.output_plot("plt1"),
)


def server(input, output, session):
    @output
    @render.plot
    def plt1():
        df_filter = df[df['a']<=input.max_a()]
        fig = plt.scatter(df_filter.a, df_filter.b)
        return fig


app = App(app_ui, server)
