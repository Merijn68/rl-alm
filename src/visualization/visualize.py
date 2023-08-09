import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Tuple

from src.data.definitions import FIGURES_PATH

# Generic setup parameters for Matplotlib
figsize = (10, 6)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["figure.figsize"] = figsize
sns.set_style("darkgrid")


def lineplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    x_label: str = "",
    y_label: str = "",
    hue: str = "",
    figsize: Tuple[int, int] = figsize,
    name: str = "lineplot",
    title: str = "",
    figurepath: Path = Path(FIGURES_PATH),
) -> plt.Axes:
    """simple line plot"""

    plt.figure(figsize=figsize)
    if hue == "":
        ax = sns.lineplot(data=data, x=x, y=y)
    else:
        ax = sns.lineplot(data=data, x=x, y=y, hue=hue)
    ax.set_title(title)
    ax.xaxis.set_major_locator(mpl.dates.YearLocator())

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")
    return ax


def barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    x_label: str = "",
    y_label: str = "",
    hue: str = "",
    figsize: Tuple[int, int] = figsize,
    name: str = "lineplot",
    title: str = "",
    figurepath: Path = Path(FIGURES_PATH),
) -> plt.Axes:
    """simple line plot"""

    plt.figure(figsize=figsize)
    if hue == "":
        ax = sns.barplot(data=data, x=x, y=y)
    else:
        ax = sns.barplot(data=data, x=x, y=y, hue=hue)
    ax.set_title(title)

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")
    return ax


def bpvplot(
    bpv: pd.DataFrame,
    limits: pd.DataFrame,
    figsize: Tuple[int, int] = figsize,
    name: str = "bpv_profile",
    title: str = "",
    figurepath: Path = Path(FIGURES_PATH),
) -> plt.Axes:
    plt.figure(figsize=figsize)

    df = pd.concat([limits.set_index("tenor"), bpv], axis=1).reset_index()
    df = df[df["tenor"] > 0]
    ax = plt.subplots()
    ax = sns.barplot(x=df["tenor"], y=df["dv01"])
    ax = sns.barplot(
        x=df["tenor"],
        y=df["upper_limit"],
        alpha=0.2,
        width=1,
        linewidth=0,
        color="steelblue",
    )
    ax = sns.barplot(
        x=df["tenor"],
        y=df["lower_limit"],
        alpha=0.2,
        width=1,
        linewidth=0,
        color="steelblue",
    )
    ax.set_title(title)
    ax.set(xlabel="tenor", ylabel="BPV profile")
    ax.grid(False)

    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")
    return ax


def curveplot(
    curve_data: np.ndarray,
    sim_data: np.ndarray,
    start_date: np.datetime64,
    figsize: Tuple[int, int] = figsize,
    name: str = "curveplot",
    figurepath: Path = Path(FIGURES_PATH),
    title: str = "Simulated Interest Rate Curves with Correlation",
):
    """Plot the simulated and original interest rate curves"""

    BANK_RATE_TENORS = [12, 60, 120, 240]
    SWAP_TENORS = [0, 3, 6, 9, 12, 15, 18, 24, 36, 48, 60, 84, 120, 180, 360]

    num_data_steps = curve_data.shape[1]
    num_sim_steps = sim_data.shape[1]
    num_steps = num_data_steps + num_sim_steps

    data_steps = np.arange(
        start_date,
        start_date + np.timedelta64(num_data_steps, "M"),
        dtype="datetime64[M]",
    )
    sim_steps = np.arange(
        start_date + np.timedelta64(num_data_steps, "M"),
        start_date + np.timedelta64(num_steps, "M"),
        dtype="datetime64[M]",
    )

    # data_steps = np.linspace(0, num_data_steps - 1, num_data_steps)
    # sim_steps = np.linspace(num_data_steps - 1, num_steps, num_sim_steps)
    plt.figure(figsize=figsize)

    for idx, tenor in enumerate(SWAP_TENORS):
        plt.plot(data_steps, curve_data[idx, :], linestyle="-", lw=0.5, color="black")

    for idx, tenor in enumerate(SWAP_TENORS):
        plt.plot(
            sim_steps,
            sim_data[idx, :],
            linestyle="-",
            lw=0.5,
            color="red",
        )
    for idx, tenor in enumerate(BANK_RATE_TENORS):
        plt.plot(
            data_steps,
            curve_data[idx + len(SWAP_TENORS), :],
            linestyle="-",
            color="blue",
            lw=0.5,
        )
    for idx, tenor in enumerate(BANK_RATE_TENORS):
        plt.plot(
            sim_steps,
            sim_data[idx + len(SWAP_TENORS), :],
            linestyle="-",
            lw=0.5,
            color="green",
        )

    plt.xlabel("Time")
    plt.ylabel("Interest Rate")

    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="black", linestyle="-", lw=0.5),
        Line2D([0], [0], color="blue", linestyle="-", lw=0.5),
        Line2D([0], [0], color="red", linestyle="-", lw=0.5),
        Line2D([0], [0], color="green", linestyle="-", lw=0.5),
    ]

    ax = plt.gca()
    ax.legend(
        custom_lines,
        ["Swap rates", "Bank Rates", "Simulated Swap Rates", "Simulated Bank Rates"],
        loc="upper right",
    )

    plt.grid(True)
    plt.title(title)
    plt.show()
    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")

    return ax
