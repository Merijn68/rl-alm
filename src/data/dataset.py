import requests
import pandas as pd
import numpy as np
import io
from src.visualization import visualize
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import BDay
from loguru import logger
from datetime import datetime
from dateutil.parser import parse


class ECBData:
    def __init__(self):
        self._url_ = "https://sdw-wsrest.ecb.europa.eu/service/data/"
        self._start_date_ = "2003-01-01"
        self._end_date_ = "2023-02-28"

    def set_period(self, start_date, end_date):
        self._start_date_ = start_date
        self._end_date_ = end_date

    def read_data(self):
        url = self._url_
        key = self._key_
        parameters = {
            "startPeriod": self._start_date_,
            "endPeriod": self._end_date_,
        }

        response = requests.get(
            url + key, params=parameters, headers={"Accept": "text/csv"}
        )
        self.df = pd.read_csv(io.StringIO(response.text))
        return response


class Interest(ECBData):
    """Read interest data from ECB Statistical Data Warehouse"""

    def __init__(self):
        super().__init__()

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

        df["fixed_period"] = df["KEY"].map(
            {
                "MIR.M.NL.B.A2CC.F.R.A.2250.EUR.N": "<= 1 year",
                "MIR.M.NL.B.A2CC.I.R.A.2250.EUR.N": "1>5 years",
                "MIR.M.NL.B.A2CC.O.R.A.2250.EUR.N": "5>10 years",
                "MIR.M.NL.B.A2CC.P.R.A.2250.EUR.N": ">10 years",
            }
        )
        df = df.filter(["TIME_PERIOD", "fixed_period", "OBS_VALUE"], axis=1)
        df.columns = ["period", "fixed_period", "rate"]
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period")
        self.df = df
        return response

    def lineplot(self):
        visualize.lineplot(
            self.df,
            x=self.df.index,
            y="rate",
            x_label="Period",
            y_label="Interest %",
            hue="fixed_period",
            title="Loans to households for house purchase with collateral (new business) - Netherlands",
        )


class Zerocurve(ECBData):
    """Read zero curve data from ECB Statistical Data Warehouse"""

    def __init__(self):
        super().__init__()

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

    def _offset_days_(self, row):
        if row.unit == "Months":
            return row.rate_dt + DateOffset(months=row.number)
        if row.unit in ("Year", "Years"):
            return row.rate_dt + DateOffset(years=row.number)

    def read_data(self) -> pd.DataFrame:
        logger.info("reading zero curve data from ESW.")
        response = super().read_data()
        df = self.df

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
        return response

    def interpolate(self, pos_date: datetime) -> pd.DataFrame:
        # interpolate zero curve
        if pos_date not in self.df.index:
            df_interpol = pd.DataFrame()
        else:
            df_date = self.df.loc[pos_date]
            df_date = df_date.set_index("value_dt")
            df_interpol = df_date[["rate"]].resample("D").mean()
            df_interpol["rate"] = df_interpol["rate"].interpolate()
        return df_interpol

    def lineplot(self):
        visualize.lineplot(
            self.df,
            x=self.df.index,
            y="rate",
            x_label="Period",
            y_label="Interest %",
            title="Zero curve, yield curve, governement bond triple A Euro Area",
        )


class Inflation(ECBData):
    """Read inflation data from ECB Statistical Data Warehouse"""

    def __init__(self):
        super().__init__()
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
        df = self.df.filter(["TIME_PERIOD", "OBS_VALUE"], axis=1)
        df.columns = ["period", "inflation"]
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period")
        self.df = df
        return response

    def lineplot(self):
        visualize.lineplot(
            self.df,
            x=self.df.index,
            y="inflation",
            x_label="Period",
            y_label="Inflation %",
            title="HICP Annual rate of change Eurostat",
        )


def main():
    zerocurve = Zerocurve()
    zerocurve.read_data()
    pos_date = parse("28-02-2023")
    df_i = zerocurve.interpolate(pos_date)


if __name__ == "__main__":
    main()

# def get_interest_data(
#     path: Path = Path("../data/raw/bonds.csv"),
# ) -> pd.DataFrame:
#     """Get interest rates dataset from DNB Statistics"""

#     logger.info("getting interest data from file ")
#     df = pd.read_excel(
#         "../data/raw/interest.xlsx", parse_dates=["Periode "], index_col="Periode "
#     )
#     df.index.rename("period", inplace=True)
#     df = df.query("Instrument == 'Woninghypotheken - Totaal ' ")
#     df = df.query("Stroomtype == 'Nieuwe contracten '")
#     df = df.query("RenteVastPeriode != 'Totaal * '")
#     df = df[["RenteVastPeriode", "waarde"]]
#     df.columns = ["fixed_period", "interest"]  # subset columns
#     df["fixed_period"] = df["fixed_period"].replace(
#         {
#             "> 1 jaar en <= 5 jaar * ": "1>5 years",
#             "> 5 jaar en <= 10 jaar * ": "5>10 years",
#             "Variabel en <= 1 jaar * ": "<= 1 year",
#             "> 10 jaar * ": "> 10 years",
#         }
#     )
#     df["fixed_period"] = df["fixed_period"].astype("category")
#     df["fixed_period"] = df["fixed_period"].cat.set_categories(
#         ["<= 1 year", "1>5 years", "5>10 years", "> 10 years"], ordered=True
#     )
#     return df
