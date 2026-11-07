"""Shared figure theme, figure saving and the SHAP force plot."""

from collections.abc import Iterable
from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"

COLORS = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "grey": "#b4b2a9"}
"""Blue = no stroke / single series, orange = stroke, grey = "without" baselines."""

FIGURE_WIDTH = 7.0
"""Report text width in inches (US Letter, 0.75 in margins)."""


def set_theme() -> None:
    """Apply the shared figure theme.

    Palette, no top/right spines, inward ticks, constrained layout (so legends
    outside the axes stay inside the figure width), and text at print size:
    titles 10 pt, axis labels 9 pt, ticks and legends 8 pt. Figures show at
    150 dpi in the notebook and are saved at 300 dpi.
    """
    plt.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=[COLORS["blue"], COLORS["orange"], COLORS["green"]]),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "boxplot.medianprops.color": "black",
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: Figure, name: str) -> None:
    """Save ``fig`` as ``figures/<name>.png``, trimmed to its content."""
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight")


def label_panels(axes: Iterable[Axes]) -> None:
    """Mark each panel of a multi-panel figure with (a), (b), ...

    The letter is prefixed to the panel title if there is one, otherwise placed in
    bold at the panel's top left.
    """
    for letter, ax in zip(ascii_lowercase, axes, strict=False):
        if ax.get_title():
            ax.set_title(f"({letter}) {ax.get_title()}")
        else:
            ax.text(
                -0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold"
            )


def force_plot(base_value: float, values: np.ndarray, features: pd.Series, title: str) -> Figure:
    """Draw a single-patient SHAP force plot.

    Only contributions of at least 10% of the total effect are labelled, to keep
    the plot readable.

    Args:
        base_value: Model output over the background (start of the plot).
        values: The patient's SHAP values.
        features: The patient's feature values, indexed by display label.
        title: Figure title.
    """
    # SHAP lays out its own text, which the theme's constrained layout would distort
    with plt.rc_context({"figure.constrained_layout.use": False}):
        shap.force_plot(
            base_value,
            values,
            features.round(1),
            feature_names=list(features.index),
            matplotlib=True,
            show=False,
            figsize=(10, 3.0),  # shorter overlaps the output value with its "f(x)" label
            text_rotation=15,
            contribution_threshold=0.1,
        )
    fig = plt.gcf()
    fig.suptitle(title, y=1.4)
    return fig
