
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Tuple


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
    title: str = "" ,
    figurepath: Path = Path("../reports/figures"),
) -> plt.Axes:
    """simple line plot"""

    plt.figure(figsize=figsize)
    if hue == '':
        ax = sns.lineplot(data=data, x=x, y=y)
    else:
        ax = sns.lineplot(data=data, x=x, y=y, hue=hue)
    ax.set_title(title)
    
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    

    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight");



    return ax
