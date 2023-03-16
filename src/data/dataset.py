import requests
import pandas as pd
import numpy as np
import io
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import BDay
from loguru import logger
from datetime import datetime
from dateutil.parser import parse
from pathlib import Path

from src.visualization import visualize
from src.models import predict
from src.data.definitions import DATA_RAW


class ECBData:
    def __init__(self):
        self._url_ = "https://sdw-wsrest.ecb.europa.eu/service/data/"
        self._start_date_ = parse("2003-01-01")
        self._end_date_ = parse("2023-02-28")
        self.df = pd.DataFrame()

    def set_period(self, start_date: datetime, end_date: datetime):
        self._start_date_ = start_date
        self._end_date_ = end_date

    def get_period(self) -> tuple[datetime, datetime]:
        return self._start_date_, self._end_date_

    def read_data(self):
        url = self._url_
        key = self._key_
        parameters = {
            "startPeriod": self._start_date_.strftime("%Y-%m-%d"),
            "endPeriod": self._end_date_.strftime("%Y-%m-%d"),
        }
        try:
            response = requests.get(
                url + key, params=parameters, headers={"Accept": "text/csv"}
            )
            self.df = pd.read_csv(io.StringIO(response.text))
        except pd.errors.EmptyDataError:
            logger.error(
                f"Dataset '{self.name}' not loaded for period {self._start_date_} to {self._end_date_}"
            )
        return response

    def load_data(self):
        data = Path(DATA_RAW, f"{self.name}.csv")
        if data.exists():
            self.df = pd.read_csv(data)
        else:
            logger.error(
                "Data not found. Use read_data to initalize data load from ECB"
            )

    def save_data(self):
        self.df.to_csv(Path(DATA_RAW, f"{self.name}.csv"))


class Interest(ECBData):
    """Read interest data from ECB Statistical Data Warehouse"""

    def __init__(self):
        super().__init__()
        self.name = "interest"

        DATAFLOW = "MIR"
        FREQ = "M"
        REF_AREA = "NL"
        SECTOR = "B"
        BALANCE_ITEM = "A2CC"
        MATURITY = "F+I+O+P"
        DATA_TYPE = "R"
        AMOUNT_CAT = "A"
        COUNTERPARTY_SECTOR = "2250"
        CURRENCY = "EUR"
        COVERAGE = "N"
        self._key_ = (
            f"{DATAFLOW}/{FREQ}.{REF_AREA}.{SECTOR}.{BALANCE_ITEM}."
            f"{MATURITY}.{DATA_TYPE}.{AMOUNT_CAT}."
            f"{COUNTERPARTY_SECTOR}.{CURRENCY}.{COVERAGE}"
        )

    def read_data(self):
        logger.info("reading interest data from ESW.")
        response = super().read_data()
        df = self.df
        if df.empty:
            return response

        df["fixed_period"] = df["KEY"].map(
            {
                "MIR.M.NL.B.A2CC.F.R.A.2250.EUR.N": "<= 1 year",
                "MIR.M.NL.B.A2CC.I.R.A.2250.EUR.N": "1>5 years",
                "MIR.M.NL.B.A2CC.O.R.A.2250.EUR.N": "5>10 years",
                "MIR.M.NL.B.A2CC.P.R.A.2250.EUR.N": ">10 years",
            }
        )
        df = df.filter(["TIME_PERIOD", "fixed_period", "OBS_VALUE"], axis=1)
        df.columns = ["period", "fixed_period", "interest"]
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period")
        df.sort_values(["period", "fixed_period"], inplace=True)
        self.df = df
        return response

    def lineplot(self):
        visualize.lineplot(
            self.df,
            x=self.df.index,
            y="interest",
            x_label="Period",
            y_label="Interest %",
            hue="fixed_period",
            title="Loans to households for house purchase with collateral (new business) - Netherlands",
        )

    def load_data(self):
        super().load_data()
        df = self.df
        df.set_index("period", inplace=True)
        df.sort_values(["period", "fixed_period"], inplace=True)
        self.df = df


class Zerocurve(ECBData):
    """Read zero curve data from ECB Statistical Data Warehouse"""

    def __init__(self):
        super().__init__()
        self.name = "zerocurve"
        DATAFLOW = "YC"  # Yield Curve
        FREQ = "B"  # Daily - buisiness days
        REF_AREA = "U2"  # Euro area (changing composition)
        CURRENCY = "EUR"  # currency
        PROVIDER_FM = "4F"  # ECB
        INSTRUMENT_FM = "G_N_A"  # Government bond, triple A
        PROVIDER_FM_ID = "SV_C_YM"  # Svensson model,continuous compounding
        DATA_TYPE_FM = (
            "IF_3M+IF_6M+IF_9M+IF_1Y+IF_1Y3M+IF_1Y6M+IF_2Y+IF_3Y+"
            "IF_4Y+IF_5Y+IF_7Y+IF_10Y+IF_15Y+IF_30Y"
        )
        self._key_ = (
            f"{DATAFLOW}/{FREQ}.{REF_AREA}.{CURRENCY}.{PROVIDER_FM}."
            f"{INSTRUMENT_FM}.{PROVIDER_FM_ID}.{DATA_TYPE_FM}"
        )
        self.timestep = 0

    def _offset_days_(self, row):
        if row.unit == "Months":
            return row.rate_dt + DateOffset(months=row.number)
        if row.unit in ("Year", "Years"):
            return row.rate_dt + DateOffset(years=row.number)

    def read_data(self) -> pd.DataFrame:
        logger.info("reading zero curve data from ESW.")
        response = super().read_data()
        df = self.df
        if df.empty:
            return response
        df.loc[:, "KEY"] = df.loc[:, "KEY"].str.slice(29)
        df = df.loc[:, ["TIME_PERIOD", "KEY", "OBS_VALUE"]]
        df.columns = ["rate_dt", "tenor", "rate"]
        names = {
            "IF_3M": "3 Months",
            "IF_6M": "6 Months",
            "IF_9M": "9 Months",
            "IF_1Y3M": "15 Months",
            "IF_1Y6M": "18 Months",
            "IF_1Y": "1 Year",
            "IF_2Y": "2 Years",
            "IF_3Y": "3 Years",
            "IF_4Y": "4 Years",
            "IF_5Y": "5 Years",
            "IF_7Y": "7 Years",
            "IF_10Y": "10 Years",
            "IF_15Y": "15 Years",
            "IF_30Y": "30 Years",
        }
        df["tenor"] = df["tenor"].map(names)
        df[["number", "unit"]] = df["tenor"].str.split(" ", expand=True)
        df["number"] = df["number"].astype("int")
        df["unit"] = df["unit"].astype("str")
        df["rate_dt"] = pd.to_datetime(df["rate_dt"])
        df["value_dt"] = df.apply(self._offset_days_, axis=1)
        df.drop(columns=["number", "unit"], inplace=True)

        # Add Overnight data point
        df_on = df[["rate_dt"]].drop_duplicates()
        df_on["rate"] = np.NaN
        df_on["tenor"] = "ON"
        df_on["value_dt"] = df_on["rate_dt"] + BDay(1)
        df = pd.concat([df.reset_index(), df_on])
        df.drop(columns=["index"], inplace=True)
        df.set_index("rate_dt", inplace=True)
        df.sort_values(["rate_dt", "value_dt"], inplace=True)
        df.bfill(inplace=True)

        self.df = df
        self.yield_data = self.df.pivot(columns="tenor", values="rate")
        self._calculate_()

        return response

    def _calculate_(self):
        """Calculate statistics (mu and sigma) for simulating rates"""
        rate_changes = np.log(1 + self.yield_data.pct_change()).dropna()
        self.mu = rate_changes.mean().values
        self.sigma = rate_changes.std().values

    def interpolate(self, pos_date: datetime) -> pd.DataFrame:
        # In order to caculate the net present value at any timestep, we need to
        # interpolate the zero curve to a forward curve for that specific position date
        # To simplify matters we use linair interpolation
        if pos_date not in self.df.index:
            logger.error("Interpolation data {pos_date} not found in zero curve data.")
        else:
            df_date = self.df.loc[pos_date]
            df_date = df_date.set_index("value_dt")
            df_forward = df_date[["rate"]].resample("D").mean()
            df_forward["rate"] = df_forward["rate"].interpolate()
            return df_forward

    def lineplot(self):
        if self.yield_data.empty:
            return
        df = pd.melt(self.yield_data.reset_index(), id_vars="rate_dt")
        visualize.lineplot(
            df,
            x="rate_dt",
            y="value",
            hue="tenor",
            title="Zero curve, yield curve, governement bond triple A Euro Area",
        )

    def step(self, dt=1 / 252):
        # Move one step forward in time, generating simulated data for one day
        yield_data = self.yield_data
        mu = self.mu
        sigma = self.sigma
        r0 = yield_data[-1].values
        r1 = np.exp(
            predict.vasicek(np.log(r0), mu, sigma, dt)
        )  # Exponentiate the predicted log return
        next_day = yield_data.index(-1) + BDay(1)
        self.yield_data.loc[next_day] = r1
        self.timestep = self.timestep + 1

    def reset(self):
        # Reset the time to the origin
        self.yield_data = self.df.pivot(columns="tenor", values="rate")


class Inflation(ECBData):
    """Read inflation data from ECB Statistical Data Warehouse"""

    def __init__(self):
        super().__init__()
        self.name = "inflation"
        DATAFLOW = "ICP"
        FREQ = "M"
        REF_AREA = "NL"
        ADJUSTMENT = "N"
        CLASS = "000000"
        PROVIDER = "4"  # ECB
        VALIDATION = "ANR"
        self._key_ = (
            f"{DATAFLOW}/{FREQ}.{REF_AREA}."
            f"{ADJUSTMENT}.{CLASS}.{PROVIDER}.{VALIDATION}"
        )

    def read_data(self):
        logger.info("reading inflation data from ESW.")
        response = super().read_data()
        df = self.df
        if df.empty:
            return response
        df = self.df.filter(["TIME_PERIOD", "OBS_VALUE"], axis=1)
        df.columns = ["period", "inflation"]
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period")
        self.df = df
        return response

    def lineplot(self):
        if self.df.empty:
            return
        visualize.lineplot(
            self.df,
            x=self.df.index,
            y="inflation",
            x_label="Period",
            y_label="Inflation %",
            title="HICP Annual rate of change Eurostat",
        )
