import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta

from src.data.definitions import FIGURES_PATH

TENORS = {
    "ON": 0,
    "IF_3M": 3,
    "IF_6M": 6,
    "IF_9M": 9,
    "IF_1Y": 12,
    "IF_1Y3M": 15,
    "IF_1Y6M": 18,
    "IF_2Y": 24,
    "IF_3Y": 36,
    "IF_4Y": 48,
    "IF_5Y": 60,
    "IF_7Y": 84,
    "IF_10Y": 120,
    "IF_15Y": 180,
    "IF_30Y": 360,
}


# Generic setup parameters for Matplotlib
FIGSIZE = (10, 6)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["figure.figsize"] = FIGSIZE
sns.set_style("darkgrid")
sns.set()


def situational_plot(
    pos_date,
    cf_proj_cashflows,
    cf_funding,
    cf_mortgages,
    interest_rates,
    zero_rates,
    mortgages,
    funding,
    num_cols: int = 2,
    num_rows: int = 2,
    title: str = "",
    figsize: Tuple[int, int] = FIGSIZE,
    figurepath: Path = Path(FIGURES_PATH),
) -> plt.Axes:
    """Plot the current state of the Bank Model"""

    fig, axes = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=figsize)
    fig.suptitle(title + " " + str(pos_date))
    ax = axes[0, 0]
    # First plot is the projected cashflows
    ax.set_title("Projected cashflows")
    years = np.arange(0, 31)
    differences = cf_mortgages["cashflow"] + cf_funding["cashflow"]
    data = {
        "Years": years,
        "Mortgages": cf_mortgages["cashflow"],
        "Funding": cf_funding["cashflow"],
        "Surplus": np.maximum(differences, 0),
        "Shortage": np.minimum(differences, 0),
    }
    sns.barplot(
        data=data, x="Years", y="Mortgages", color="skyblue", label="Mortgages", ax=ax
    )
    sns.barplot(
        data=data, x="Years", y="Funding", color="coral", label="Funding", ax=ax
    )
    sns.barplot(
        data=data,
        x="Years",
        y="Surplus",
        color="lime",
        label="Surplus",
        ax=ax,
    )
    sns.barplot(
        data=data,
        x="Years",
        y="Shortage",
        color="red",
        label="Shortage",
        ax=ax,
    )
    ax.set_xlabel("Years")
    ax.set_ylabel("Amount")
    ax.legend()

    # Second plot is the zero rates per tenor
    ax = axes[0, 1]
    start_date = pos_date
    dates = [
        start_date + np.array(tenor, "timedelta64[M]") for tenor in TENORS.values()
    ]

    ax.set_title("Zero rates")
    sns.lineplot(
        x=dates,
        y=zero_rates[:, 0],
        ax=ax,
    )
    ax.set_xlabel("Tenors")
    ax.set_ylabel("Rates")

    ax = axes[1, 0]
    ax.set_title("Mortgages outstanding")
    data = {"Tenor": mortgages["tenor"], "Principal": mortgages["principal"]}
    df = pd.DataFrame(data)
    total_per_tenor = df.groupby("Tenor")["Principal"].sum().reset_index()
    sns.barplot(ax=ax, x="Tenor", y="Principal", data=total_per_tenor)

    ax = axes[1, 1]
    ax.set_title("Funding outstanding")
    data = {"Tenor": funding["tenor"], "Principal": funding["principal"] * -1}
    df = pd.DataFrame(data)
    total_per_tenor = df.groupby("Tenor")["Principal"].sum().reset_index()
    sns.barplot(ax=ax, x="Tenor", y="Principal", data=total_per_tenor)
    min_value = 0
    max_value = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 0].set_ylim(min_value, max_value)
    axes[1, 1].set_ylim(min_value, max_value)

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()


def plot_rewards(
    episode_rewards,
    episode_nii,
    episode_risk_penalty,
    episode_liquidity_penalty,
    figsize: Tuple[int, int] = FIGSIZE,
    name: str = "rewards",
    figurepath: Path = Path(FIGURES_PATH),
) -> plt.Axes:
    plt.title(f"Rewards for {len(episode_rewards)} episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.xlim([0, len(episode_rewards)])
    all_range = np.concatenate(
        [
            episode_rewards,
            episode_nii,
            episode_risk_penalty,
            episode_liquidity_penalty,
        ]
    )
    plt.ylim(
        [
            min(all_range),
            max(all_range),
        ]
    )
    plt.plot(np.array(episode_rewards), label="Reward")
    # plt.plot(np.array(episode_nii), label="NII")
    # plt.plot(np.array(episode_risk_penalty), label="Risk Penalty")
    # plt.plot(np.array(episode_liquidity_penalty), label="Liquidity")
    plt.legend()
    plt.show()


def lineplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    x_label: str = "",
    y_label: str = "",
    hue: str = "",
    figsize: Tuple[int, int] = FIGSIZE,
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
    figsize: Tuple[int, int] = FIGSIZE,
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
    figsize: Tuple[int, int] = FIGSIZE,
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
    figsize: Tuple[int, int] = FIGSIZE,
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
    if num_sim_steps > len(sim_steps):  # This may vary with leap years
        sim_steps = np.append(sim_steps, sim_steps[-1] + np.timedelta64(1, "M"))

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
