import pandas as pd

from src.multipanneled_app.plotting import PlotConfig, create_plot


def test_create_scatter_plot():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6], "g": ["a", "a", "b"]})
    fig, ax = create_plot(df, PlotConfig(plot_type="scatter", x="x", y="y", hue="g"))
    assert fig is not None
    assert ax.get_xlabel() == "x"
