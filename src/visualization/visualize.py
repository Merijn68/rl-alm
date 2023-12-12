import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy import interpolate

from src.data.definitions import FIGURES_PATH
from matplotlib.lines import Line2D

TENORS = {
    "Overnight": 0,
    "3 Months": 3,
    "6 Months": 6,
    "9 Months": 9,
    "1 Year": 12,
    "15 Months": 15,
    "18 months": 18,
    "2 Years": 24,
    "3 Years": 36,
    "4 Years": 48,
    "5 Years": 60,
    "7 Years": 84,
    "10 Years": 120,
    "15 Years": 180,
    "30 Years": 360,
}

YEARS = [1, 5, 10, 20]

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
    timestep,
    liquidity,
    reward,
    risk_penalty,
    cf_proj_cashflows,
    cf_funding,
    cf_mortgages,
    interest_rates,
    zero_rates,
    mortgages,
    funding,
    num_cols: int = 2,
    num_rows: int = 1,
    title: str = "",
    figsize: Tuple[int, int] = (20, 6),
    figurepath: Path = Path(FIGURES_PATH),
) -> plt.Axes:
    """Plot the current state of the Bank Model"""

    name = "situational_plot_" + str(timestep)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        ncols=num_cols,
        nrows=num_rows,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"wspace": 0.2, "hspace": 0.2},
    )
    fig.suptitle(title + " " + str(pos_date))
    ax = axes[0, 1]

    ylim = 25000

    # First plot is the outstanding mortgages and funding, and highlight the new mortgages and funding
    ax.set_title("Mortgages and Funding")
    df_m = pd.DataFrame(mortgages)
    df_m = df_m[df_m["maturity_date"] > pos_date]
    df_f = pd.DataFrame(funding)
    df_f["principal"] = df_f["principal"] * -1
    df_f = df_f[df_f["maturity_date"] > pos_date]

    # Filter data for the latest month
    latest_month_data = df_m[(df_m["start_date"] >= pd.to_datetime(pos_date))]

    bar_width = 0.4
    bar_centers = np.arange(0, len(YEARS))

    d = df_m.groupby("tenor")["principal"].sum().reindex(YEARS, fill_value=0)
    ax.bar(
        bar_centers,
        d,
        alpha=0.4,
        color="blue",
        width=bar_width,
    )

    dn = (
        latest_month_data.groupby("tenor")["principal"]
        .sum()
        .reindex(YEARS, fill_value=0)
    )
    bottom = d - dn
    ax.bar(
        bar_centers,
        dn,
        alpha=0.6,
        color="blue",
        width=bar_width,
        bottom=bottom,
    )

    d = df_f.groupby("tenor")["principal"].sum().reindex(YEARS, fill_value=0)
    ax.bar(bar_centers + bar_width, d, alpha=0.4, color="red", width=bar_width)
    latest_month_data = df_f[
        (df_f["start_date"].dt.month == pd.to_datetime(pos_date).month)
    ]
    dn = (
        latest_month_data.groupby("tenor")["principal"]
        .sum()
        .reindex(YEARS, fill_value=0)
    )
    bottom = d - dn
    ax.bar(
        bar_centers + bar_width,
        dn,
        alpha=0.6,
        color="red",
        width=bar_width,
        bottom=bottom,
    )

    ax.set_xlabel("Years")
    ax.set_ylabel("Amount")
    ax.set_xticks(bar_centers, YEARS)
    ax.set_ylim(0, ylim)

    # Add a text column to the right of the plot
    h_offset = 1.3

    l_column, r_column = 15, 6
    plt.text(
        h_offset,
        1,
        f"Time Step:".ljust(l_column)
        + f"{timestep}".rjust(r_column)
        + "\n"
        + "\n"
        + f"Reward:".ljust(l_column)
        + f"{reward:0.0f}".rjust(r_column)
        + "\n"
        + f"Penalty:".ljust(l_column)
        + f"{risk_penalty:0.0f}".rjust(r_column),
        fontsize=10,
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    plt.text(
        h_offset,
        0.80,
        f"Liquidity:".ljust(l_column) + f"{liquidity:0.0f}".rjust(r_column),
        fontsize=10,
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    bankrate0 = interest_rates[0].item()
    bankrate1 = interest_rates[1].item()
    bankrate2 = interest_rates[2].item()
    bankrate3 = interest_rates[3].item()

    plt.text(
        h_offset,
        0.50,
        f"Bank Rates".ljust(l_column)
        + "\n"
        + f"<= 1 year".ljust(l_column)
        + f"{bankrate0:0.02f}".rjust(r_column)
        + "\n"
        + f"<= 5 year".ljust(l_column)
        + f"{bankrate1:0.02f}".rjust(r_column)
        + "\n"
        + f"<= 10 year".ljust(l_column)
        + f"{bankrate2:0.02f}".rjust(r_column)
        + "\n"
        + f"> 10 year".ljust(l_column)
        + f"{bankrate3:0.02f}".rjust(r_column)
        + "\n",
        fontsize=10,
        transform=ax.transAxes,
        ha="right",
    )

    plt.text(
        h_offset,
        0.43,
        f"Zero Rates".ljust(l_column) + "\n",
        fontsize=10,
        transform=ax.transAxes,
        ha="right",
    )

    for index, key in enumerate(TENORS):
        plt.text(
            h_offset,
            0.40 - index * 0.03,
            f"{key}".ljust(l_column)
            + f"{zero_rates[index].item():0.02f}".rjust(r_column)
            + "\n",
            fontsize=10,
            transform=ax.transAxes,
            ha="right",
        )

    custom_lines = [
        Line2D([0], [0], color="blue", alpha=0.4, lw=4),
        Line2D([0], [0], color="blue", alpha=0.6, lw=4),
        Line2D([0], [0], color="red", alpha=0.4, lw=4),
        Line2D([0], [0], color="red", alpha=0.6, lw=4),
    ]

    ax.legend(
        custom_lines,
        ["Mortgages", "Mortgages (New Business)", "Funding", "Funding (New Issue)"],
    )

    ax = axes[0, 0]

    # second plot is the projected netted cashflows
    ax.set_title("Net Projected cashflows")
    cf_mortgages = cf_mortgages[0:21]
    cf_funding = cf_funding[0:21]
    years = np.arange(0, 21)
    differences = cf_mortgages["cashflow"] + cf_funding["cashflow"]
    data = {
        "Years": years,
        "Surplus": np.maximum(differences, 0),
        "Shortage": np.minimum(differences, 0),
    }
    ax.axhline(y=6000, color="r", linestyle="--", label="Threshold")
    ax.axhline(y=-6000, color="r", linestyle="--")
    sns.barplot(
        data=data,
        x="Years",
        y="Surplus",
        color="blue",
        label="Net Cashflow",
        ax=ax,
    )
    sns.barplot(
        data=data,
        x="Years",
        y="Shortage",
        color="blue",
        ax=ax,
    )
    ax.set_xlabel("Years")
    ax.set_ylabel("Amount")
    ax.legend()

    ax.set_ylabel("Amount")
    ax.set_ylim(-10000, 10000)

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")
    plt.show()


# def situational_plot(
#     pos_date,
#     cf_proj_cashflows,
#     cf_funding,
#     cf_mortgages,
#     interest_rates,
#     zero_rates,
#     mortgages,
#     funding,
#     num_cols: int = 2,
#     num_rows: int = 2,
#     title: str = "",
#     figsize: Tuple[int, int] = FIGSIZE,
#     figurepath: Path = Path(FIGURES_PATH),
# ) -> plt.Axes:
#     """Plot the current state of the Bank Model"""

#     fig, axes = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=figsize)
#     fig.suptitle(title + " " + str(pos_date))

#     ax = axes[0, 0]
#     # First plot is the projected netted cashflows
#     ax.set_title("Net Projected cashflows")
#     years = np.arange(0, 31)
#     differences = cf_mortgages["cashflow"] + cf_funding["cashflow"]
#     data = {
#         "Years": years,
#         "Surplus": np.maximum(differences, 0),
#         "Shortage": np.minimum(differences, 0),
#     }
#     ax.axhline(y=5000, color="r", linestyle="--", label="Threshold")
#     ax.axhline(y=-5000, color="r", linestyle="--")
#     sns.barplot(
#         data=data,
#         x="Years",
#         y="Surplus",
#         color="blue",
#         label="Net Cashflow",
#         ax=ax,
#     )
#     sns.barplot(
#         data=data,
#         x="Years",
#         y="Shortage",
#         color="blue",
#         ax=ax,
#     )
#     ax.set_xlabel("Years")
#     ax.set_ylabel("Amount")

#     ax.legend()

#     # Second plot is the zero rates per tenor
#     ax = axes[0, 1]
#     start_date = pos_date
#     dates = [
#         start_date + np.array(tenor, "timedelta64[M]") for tenor in TENORS.values()
#     ]

#     ax.set_title("Zero rates")
#     sns.lineplot(
#         x=dates,
#         y=zero_rates[:, 0],
#         ax=ax,
#     )

#     ax.set_xlabel("Tenors")
#     ax.set_ylabel("Rates")

#     ax = axes[1, 0]
#     ax.set_title("Outstanding Mortgages and Bonds")
#     m = {
#         "tenor": mortgages["tenor"],
#         "principal": mortgages["principal"],
#         "type": "Mortgages",
#     }
#     f = {
#         "tenor": funding["tenor"],
#         "principal": funding["principal"] * -1,
#         "type": "Funding",
#     }
#     df = pd.concat([pd.DataFrame(m), pd.DataFrame(f)])

#     df_f = pd.DataFrame(funding)
#     df_m = pd.DataFrame(mortgages)
#     filter1 = df_f["start_date"] == df_f["start_date"].max()
#     filter2 = df_m["start_date"] == df_m["start_date"].max()
#     df_f = df_f.where(filter1)
#     df_m = df_m.where(filter2)

#     total_per_tenor = df.groupby(["type", "tenor"])["principal"].sum().reset_index()
#     sns.barplot(ax=ax, x="tenor", y="principal", hue="type", data=total_per_tenor)
#     sns.barplot(
#         ax=ax,
#         x="tenor",
#         y="principal",
#         hue="type",
#         data=pd.concat([df_f, df_m]),
#         alpha=0.2,
#     )

#     ax = axes[1, 1]
#     ax.clear()
#     # Remove spines and ticks
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["bottom"].set_visible(False)
#     ax.spines["left"].set_visible(False)
#     ax.tick_params(top=False, bottom=False, left=False, right=False)
#     ax.xaxis.set_ticks([])  # Hide x-axis ticks
#     ax.yaxis.set_ticks([])  # Hide y-axis ticks
#     ax.set_facecolor("none")  # Set background color to be transparent

#     # Calculate the differences
#     df_wide = (
#         df.groupby(["tenor", "type"])["principal"]
#         .sum()
#         .unstack()
#         .reset_index()
#         .fillna(0)
#     )
#     df_wide["difference"] = df_wide["Mortgages"] - df_wide["Funding"]

#     # Calculate total differences
#     total_differences = df_wide["difference"].sum()

#     # Construct table data
#     table_data = [
#         ["Total Mortgages", sum(mortgages["principal"])],
#         ["Total Funding", sum(funding["principal"])],
#         ["Total Difference", total_differences],
#     ]


#     table = ax.table(cellText=table_data, loc="center", cellLoc="left")
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1.2, 1.2)

#     plt.subplots_adjust(hspace=0.5)
#     plt.tight_layout()
#     plt.show()


# def plot_funding(
#     sa_funding: np.ndarray,
#     figsize: Tuple[int, int] = FIGSIZE,
#     name: str = "funding",
#     figurepath: Path = Path(FIGURES_PATH),
# ):
#     """plot funding attracted over time"""
#     plt.figure(figsize=figsize)
#     plt.title(f"Funding attracted over time")
#     plt.xlabel("time")
#     plt.ylabel("Funding")
#     plt.ylim(0, 30)
#     data = pd.DataFrame(sa_funding)
#     data["principal"] = data["principal"] * -1
#     data["year"] = data["period"].dt.year
#     data_expanded = data.loc[data.index.repeat(data["principal"])]
#     sns.violinplot(
#         data=data_expanded,
#         x="year",
#         y="tenor",
#         palette="Set2",
#         inner="quart",
#         inner_kws=dict(box_width=15, whis_width=2, color=".8"),
#     )

#     plt.show()


def plot_rewards(
    episode_rewards,
    interpolate_line=True,
    interpolate_points=100,
    figsize: Tuple[int, int] = FIGSIZE,
    ylim: Tuple[int, int] = (-2000, 2000),
    rolling_line: bool = False,
    color="red",
    label="Rewards",
    title: str = f"Rewards per episode",
    name: str = "rewards",
    figurepath: Path = Path(FIGURES_PATH),
):
    """plot episode rewards"""
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.xlim([0, len(episode_rewards)])
    plt.ylim(ylim)
    plt.plot(np.array(episode_rewards), color=color, alpha=0.2)
    if interpolate_line:
        x = np.arange(len(episode_rewards))
        bspline = interpolate.make_interp_spline(x, episode_rewards)
        x_new = np.linspace(0, len(episode_rewards) - 1, interpolate_points)
        y_new = bspline(x_new)
        plt.plot(x_new, y_new, label=label, color=color)
    if rolling_line:
        x = np.arange(len(episode_rewards))
        y = pd.DataFrame(episode_rewards).rolling(12).mean()
        plt.plot(
            x,
            y,
            color=color,
            linestyle="-",
            label=label,
        )

    plt.legend()
    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")
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
        start_date + relativedelta(months=num_data_steps),
        dtype="datetime64[M]",
    )
    sim_steps = np.arange(
        start_date + relativedelta(months=num_data_steps),
        start_date + relativedelta(months=num_steps),
        dtype="datetime64[M]",
    )
    if num_sim_steps > len(sim_steps):  # This may vary with leap years
        sim_steps = np.append(sim_steps, sim_steps[-1] + relativedelta(months=1))

    plt.figure(figsize=figsize)

    for idx, _ in enumerate(SWAP_TENORS):
        alpha = (idx + 1) / len(SWAP_TENORS)
        plt.plot(
            data_steps,
            curve_data[idx, :],
            linestyle="-",
            lw=0.8,
            color="black",
            alpha=alpha,
        )

    for idx, _ in enumerate(SWAP_TENORS):
        alpha = (idx + 1) / len(SWAP_TENORS)
        plt.plot(
            sim_steps,
            sim_data[idx, :],
            linestyle="-",
            lw=0.8,
            color="red",
            alpha=alpha,
        )
    for idx, _ in enumerate(BANK_RATE_TENORS):
        plt.plot(
            data_steps,
            curve_data[idx + len(SWAP_TENORS), :],
            linestyle="-",
            color="blue",
            lw=0.8,
        )
    for idx, _ in enumerate(BANK_RATE_TENORS):
        plt.plot(
            sim_steps,
            sim_data[idx + len(SWAP_TENORS), :],
            linestyle="-",
            lw=0.8,
            color="green",
        )

    plt.xlabel("Time")
    plt.ylabel("Interest Rate")

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

    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")
    plt.show()

    return ax


def plot_frame(f: pd.DataFrame, m: pd.DataFrame, ax):
    """Plot a Frame of the video of the bank model"""
    return ax
