import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import BDay
from loguru import logger
from datetime import datetime
from src.visualization import visualize
from src.models import predict
from src.data.definitions import DATA_RAW
from src.data import dataset

DATAFLOW = "YC"  # Yield Curve
FREQ = "B"  # Daily - buisiness days
REF_AREA = "U2"  # Euro area (changing composition)
CURRENCY = "EUR"  # currency
PROVIDER_FM = "4F"  # ECB
INSTRUMENT_FM = "G_N_A"  # Government bond, triple A
PROVIDER_FM_ID = "SV_C_YM"  # Svensson model,continuous compounding
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


class Zerocurve(dataset.ECBData):
    """Read zero curve data from ECB Statistical Data Warehouse"""

    def __init__(self):
        super().__init__()
        self.name = "zerocurve"
        data_type_fm = "+".join(list(TENORS.keys())[1:])
        self._key_ = (
            f"{DATAFLOW}/{FREQ}.{REF_AREA}.{CURRENCY}.{PROVIDER_FM}."
            f"{INSTRUMENT_FM}.{PROVIDER_FM_ID}.{data_type_fm}"
        )

    def read_data(self) -> pd.DataFrame:
        response = super().read_data()
        df = self.df
        if df.empty:
            return response
        df.loc[:, "KEY"] = df.loc[:, "KEY"].str.slice(29)
        df = df.loc[:, ["TIME_PERIOD", "KEY", "OBS_VALUE"]]
        df.columns = ["rate_dt", "tenor", "rate"]
        df["tenor"] = df["tenor"].map(TENORS)
        df["rate_dt"] = pd.to_datetime(df["rate_dt"])
        df["value_dt"] = df["rate_dt"] + df["tenor"].astype("timedelta64[M]")
        # Add Overnight data point
        df_on = df[["rate_dt"]].drop_duplicates()
        df_on["rate"] = np.NaN
        df_on["tenor"] = 0
        df_on["value_dt"] = df_on["rate_dt"] + BDay(1)
        df = pd.concat([df.reset_index(), df_on])
        df.drop(columns=["index"], inplace=True)
        df.set_index("rate_dt", inplace=True)
        df.sort_values(["rate_dt", "value_dt"], inplace=True)
        df.bfill(inplace=True)
        self.df = df
        self.origin = self.df.copy()  # keeping to many copies of this.. Refactor
        self.reset()
        return response

    def interpolate(self, pos_date: datetime) -> pd.DataFrame:
        # In order to caculate the net present value at any timestep, we need to
        # interpolate the zero curve to a forward curve for that specific position date
        # To simplify matters we use linair interpolation
        if pos_date not in self.df.index:
            logger.error(f"Interpolation data {pos_date} not found in zero curve data.")
        else:
            df_date = self.df.loc[pos_date]
            df_date = df_date.set_index("value_dt")
            df_forward = df_date[["rate"]].resample("D").mean()
            df_forward["rate"] = df_forward["rate"].interpolate()
            return df_forward

    def lineplot(self):
        if self.df.empty:
            logger.log("No data to plot.")
            return
        visualize.lineplot(
            self.df,
            x="rate_dt",
            y="value",
            hue="tenor",
            title="Zero curve, yield curve, governement bond triple A Euro Area",
        )

    def load_data(self):
        super().load_data()
        df = self.df
        df["rate_dt"] = pd.to_datetime(df["rate_dt"])
        df["value_dt"] = pd.to_datetime(df["value_dt"])
        df.set_index("rate_dt", inplace=True)
        df.sort_values(["rate_dt", "value_dt"], inplace=True)
        self.df = df
        self.origin = df.copy()
        self.reset()

    def step(self, dt=1 / 252):
        # Move one step forward in time, generating simulated data for one day
        # To slow. Need to do this without all the transforms
        yield_data = self.yield_data
        mu = self.mu
        sigma = self.sigma
        r0 = yield_data.iloc[-1].values
        r1 = np.exp(
            predict.vasicek(np.log(r0), mu, sigma, dt)
        )  # Exponentiate the predicted log return
        last_day = yield_data.index[-1]
        next_day = last_day + BDay(1)
        logger.debug(f"Stepping in zerocurve {next_day}.")
        yield_data.loc[next_day] = r1
        # update zerocurve dataframe - refactor this
        yd = yield_data.iloc[-1].reset_index()
        yd.columns = ["tenor", "rate"]
        yd["rate_dt"] = next_day
        yd["value_dt"] = yd["rate_dt"] + yd["tenor"].astype("timedelta64[M]")
        yd.set_index("rate_dt", inplace=True)
        self.df = pd.concat([self.df, yd], axis=0)

    def reset(self):
        # Reset the time to the origin
        self.df = self.origin.copy()
        self.yield_data = self.df.pivot(columns="tenor", values="rate")
        yield_data = self.df.pivot(columns="tenor", values="rate")
        df_rate_changes = yield_data.pct_change().dropna()
        df_log_rate_changes = np.log1p(df_rate_changes)
        self.mu = df_log_rate_changes.mean().values
        self.sigma = df_log_rate_changes.std().values
