# Bank Assets
import pandas as pd
import numpy as np
import datetime
from loguru import logger
from pandas.tseries.offsets import BDay
from pandas.tseries.offsets import DateOffset
from datetime import timedelta
from random import randrange
from dateutil.relativedelta import relativedelta
from src.models import predict
from src.visualization import visualize
from src.data import dataset
from src.data.zerocurve import Zerocurve
from src.data.interest import Interest

MORTGAGE_AMOUNT = 200000  # fixed amount for now


class Bankmodel:
    """Cashflow model of a bank"""

    def __init__(self, pos_date: datetime):
        self.df_cashflows = pd.DataFrame()
        self.df_mortgages = pd.DataFrame()
        self.pos_date = pos_date
        self.origin_pos_date = pos_date

    def _random_date_(self, start: datetime, end: datetime) -> datetime:
        """return a random business date between start and end date"""
        delta = end - start
        date = start + datetime.timedelta(days=randrange(delta.days))
        # shift dates to next business day if closed
        if date.weekday() >= 5:
            date = date + BDay(1)
            date = date.to_pydatetime()
        return date

    def _date_schedule_(self, start_date: str, n_periods: int) -> np.array:
        """Generate monthly payment dates schedule"""
        # create a date range with monthly frequency starting at start_date
        dates = pd.date_range(
            start=start_date, periods=n_periods, freq=DateOffset(months=1)
        )
        # shift dates to next business day if closed
        return dates.map(lambda x: x + BDay(1) if x.weekday() >= 5 else x)

    def _generate_mortgage_cashflow_(
        self, principal, interest_rate, years, fixed_period
    ):
        n = years * 12  # number of mortgage payments
        periods = fixed_period * 12
        principal_payment = principal / n  # monthly principal payment
        r = interest_rate / 12  # monthly interest rate
        cashflow = []
        cashflow.append((0, -principal))
        for i in range(1, periods + 1):
            interest_paid = principal * r
            if i == periods:
                principal_payment = principal
            principal = principal - principal_payment
            cashflow.append((i, principal_payment + interest_paid))
        cashflow_df = pd.DataFrame(cashflow, columns=["period", "cashflow"])
        return cashflow_df

    def _generate_mortgage_cashflows_(self, contracts: pd.DataFrame) -> pd.DataFrame:
        mortgage_tenor = 30
        dfs = []
        for _, row in contracts.iterrows():
            cf = self._generate_mortgage_cashflow_(
                row.principal, row.interest / 100, mortgage_tenor, row.years
            )
            cf["value_dt"] = self._date_schedule_(row.start_date, len(cf))
            cf["contract"] = row.contract
            dfs.append(cf)
        df_all = pd.concat([dfs[i] for i in range(len(contracts))], axis=0)
        return df_all

    def generate_mortgage_contracts(self, n: int, df_i: pd.DataFrame):
        """Generate a portfolio of mortgages"""
        # Generate mortgage portfolio of n mortgages, active at pos_date
        # probability of 10 years contracts is 10x higher then 1 year contract
        # as they can start 10 years before.
        # Devision in fixed interest period is dependend on the coupon rates
        category = np.random.choice(a=[0, 1, 2, 3], size=n, p=[0.03, 0.14, 0.23, 0.60])
        df = pd.DataFrame()
        df["category"] = category
        d = {0: "<= 1 year", 1: "1>5 years", 2: "5>10 years", 3: ">10 years"}
        df["fixed_period"] = df["category"].map(d).astype("category")
        df["fixed_period"].cat.set_categories(
            ["<= 1 year", "1>5 years", "5>10 years", ">10 years"], ordered=True
        )
        df["years"] = df["category"].map({0: 1, 1: 5, 2: 10, 3: 20})
        df["start_date"] = df.apply(
            lambda row: self._random_date_(
                self.pos_date - relativedelta(years=row.years), self.pos_date
            ),
            axis=1,
        )
        df["principal"] = MORTGAGE_AMOUNT 
        df["period"] = (
            df["start_date"].to_numpy().astype("datetime64[M]")
        )  # trick to get 1th of the month
        df = df.merge(df_i.reset_index(), how="left", on=["period", "fixed_period"])
        df["interest"] = df["interest"].fillna(
            df_i["interest"].iloc[-1]
        )  # fill missing values with last coupon rate - not a nice solution
        df["contract"] = np.arange(len(df))  # assign a contract id

        self.df_mortgages = pd.concat([self.df_mortgages, df])
        logger.info(f"Added {len(df)} mortgages to our portfolio.")

        df_cashflows = self._generate_mortgage_cashflows_(df)
        self.df_cashflows = pd.concat([self.df_cashflows, df_cashflows])
        logger.info(f"Added {len(df_cashflows)} cashflows to our model.")

        return df

    def reset(self):
        # Reset should reset to initial position - not just wipe them out...
        self.pos_date = self.origin_pos_date
        # self.df_cashflows = pd.DataFrame()
        # self.df_mortgages = pd.DataFrame()

    def calculate_returns(self, zerocurve: Zerocurve):
        """Calculate the NPV of the cashflows given the zero curve"""
        pos_date = self.pos_date
        df_forward = zerocurve.interpolate(pos_date)
        df_c = self.df_cashflows
        df_npv = df_c.merge(df_forward, left_on="value_dt", right_on=df_forward.index)
        df_npv["pos_dt"] = pos_date
        df_npv["year_frac"] = round(
            (df_npv["value_dt"] - df_npv["pos_dt"]) / timedelta(365, 0, 0, 0), 5
        )
        df_npv["df"] = 1 / (1 + df_npv["rate"] / 100) ** df_npv["year_frac"]
        df_npv["npv"] = round(df_npv["cashflow"] * df_npv["df"], 2)
        return df_npv["npv"].sum()

    def plot_contracts(self):
        """Simple stacked barplot of outstanding contracts"""
        df = self.df_mortgages
        if len(df):
            return (
                df.sort_values(["start_date", "fixed_period"])
                .pivot_table(
                    index=df["start_date"].dt.year,
                    columns=["fixed_period"],
                    values="principal",
                    aggfunc="count",
                    sort=False,
                )
                .plot(kind="bar", stacked=True)
            )
        else:
            return

    def plot_cashflows(self):
        """Simple plot of outstanding cashflows from position date"""
        # Cut off all cashflows prior to position date
        df = self.df_cashflows
        df_c = df[df["value_dt"] > self.pos_date]
        df_show = df_c[["cashflow"]].groupby(df_c["value_dt"].dt.strftime("%Y")).sum()
        df_show["cashflow"] = df_show["cashflow"] / 1000
        ax = visualize.barplot(
            df_show,
            x=df_show.index,
            y="cashflow",
            x_label="year",
            y_label="amount (x 1000)",
            title="sum of cashflows per year",
        )
        # using format string '{:.0f}' here but you can choose others
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(["{:0.0f}".format(x) for x in ax.get_yticks().tolist()])

    def step(self):
        self.pos_date = self.pos_date + BDay(1)
