# Bank Assets
import pandas as pd
import numpy as np
import datetime
import warnings
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
import math

MARGIN = 0  # 0.015
FLOAT_FREQ_PER_YEAR = 2
DAYS_OFFSET = BDay(0)


class Bankmodel:
    """Cashflow model of a bank"""

    def __init__(self, pos_date: datetime):
        self.df_cashflows = pd.DataFrame()
        self.df_mortgages = pd.DataFrame()
        self.df_swaps = pd.DataFrame()
        self.nmd_pricipal = 0
        self.nmd_core = 1
        self.nmd_maturity = 0

        # self.df_ratesets = pd.DataFrame()
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

    def _date_schedule_(
        self, start_date: str, n_periods: int, offset: str = "months"
    ) -> np.array:
        """Generate monthly payment dates schedule"""
        # create a date range with monthly frequency starting at start_date
        offset = offset.lower()
        if offset == "months":
            freq = DateOffset(months=1)
        elif offset == "semi annual":
            freq = DateOffset(months=6)
        else:
            freq = DateOffset(years=1)

        dates = pd.date_range(start=start_date, periods=n_periods, freq=freq)
        # shift dates to next business day if closed
        dates = dates.map(lambda x: x + BDay(1) if x.weekday() >= 5 else x)
        return dates

    def _generate_mortgage_cashflow_(
        self, principal, interest_rate, years, fixed_period
    ):
        n = years * 12  # number of mortgage payments
        periods = fixed_period * 12
        principal_payment = round(principal / n)  # monthly principal payment
        r = interest_rate / 12  # monthly interest rate
        cashflow = []
        cashflow.append((0, -principal))
        for i in range(1, periods + 1):
            interest_paid = round(principal * r, 0)
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
            df_cashflows = self._generate_mortgage_cashflow_(
                row.principal, row.interest / 100, mortgage_tenor, row.years
            )
            df_cashflows["value_dt"] = self._date_schedule_(
                row.start_date, len(df_cashflows), offset="months"
            )
            df_cashflows["contract_no"] = row.contract_no
            df_cashflows["type"] = "mortgage"
            df_cashflows["rate_type"] = "fixed"
            dfs.append(df_cashflows)
        df_all = pd.concat([dfs[i] for i in range(len(contracts))], axis=0)
        return df_all

    def generate_mortgage_contracts(
        self, n: int, df_i: pd.DataFrame, amount: float = 100000
    ):
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
        df["principal"] = amount
        df["period"] = (
            df["start_date"].to_numpy().astype("datetime64[M]")
        )  # trick to get 1th of the month
        df = df.merge(df_i.reset_index(), how="left", on=["period", "fixed_period"])
        df["interest"] = df["interest"].fillna(
            df_i["interest"].iloc[-1]
        )  # fill missing values with last coupon rate - not a nice solution
        df["contract_no"] = np.arange(len(df))  # assign a contract id

        self.df_mortgages = pd.concat([self.df_mortgages, df])
        logger.info(f"Added {len(df)} mortgages to our portfolio.")

        df_cashflows = self._generate_mortgage_cashflows_(df)
        self.df_cashflows = pd.concat([self.df_cashflows, df_cashflows])
        logger.info(f"Added {len(df_cashflows)} cashflows to our model.")

        return df

    def _generate_swap_cashflows_leg(self, swap, fixed_float, leg, freq):
        """Generate the cashflows for one leg of the swap"""
        tenor = swap.loc[0, "tenor"]
        contract_no = swap.loc[0, "contract_no"]
        start_date = swap.loc[0, "start_date"]
        principal = swap.loc[0, "principal"]
        base_rate = swap.loc[0, "base_rate"]
        margin = swap.loc[0, "margin"]
        freq_per_year = get_freq_per_year(freq)
        periods = int(freq_per_year * (tenor / 12))
        cashflow = []
        rate = None
        interest = None
        for i in range(1, periods + 1):
            if fixed_float == "fixed":
                rate = (base_rate + margin) / freq_per_year
                interest = round((principal * rate / 100 * leg) / freq_per_year, 0)
            cashflow.append((i, interest))
        df = pd.DataFrame(cashflow, columns=["period", "cashflow"])
        df["rate_type"] = fixed_float
        df["fixed_rate"] = rate
        df["leg"] = leg
        first_payment_date = start_date + pd.DateOffset(months=12 / freq_per_year)
        #       if first_payment_date.weekday() >= 5:
        #            first_payment_date = first_payment_date + BDay(1)
        #            first_payment_date = first_payment_date.to_pydatetime()
        df["value_dt"] = self._date_schedule_(first_payment_date, len(df), offset=freq)
        df["rateset_dt"] = pd.to_datetime(
            np.where(
                df["rate_type"] == "float",
                self._date_schedule_(start_date, len(df), offset=freq),
                None,
            )
        )
        df["contract_no"] = contract_no
        df["type"] = "swap"
        df["principal"] = principal
        return df

    def _generate_swap_cashflows_(
        self, swap: pd.DataFrame, zerocurve: Zerocurve
    ) -> pd.DataFrame:
        """Generate cashflows for an interest rate swap"""

        pay = swap.loc[0, "pay"]
        rec = swap.loc[0, "rec"]
        pay_freq = swap.loc[0, "pay_freq"]
        rec_freq = swap.loc[0, "rec_freq"]
        PAY_LEG = -1
        REC_LEG = 1
        if pay == "float":
            df_float = self._generate_swap_cashflows_leg(swap, pay, PAY_LEG, pay_freq)
        else:
            df_float = self._generate_swap_cashflows_leg(swap, rec, REC_LEG, rec_freq)
        df_npv = self.calculate_npv(zerocurve, df_float, self.pos_date)

        # Theoretical swap price calculation
        # To calculate the swap price, I assume that all floaters are 6m and all fixed rates annual.
        # Thus - taking every second row of the floating discount factors gives me the fixed leg discount factors.
        df_annualized = df_npv[1::2].copy()
        avg_df = df_annualized["df"].mean()
        swap.base_rate = abs(
            df_npv["npv"].sum()
            / avg_df
            / len(df_annualized.index)
            / swap["principal"]
            * 100
        )
        df_cashflows = pd.DataFrame()
        df_pay = self._generate_swap_cashflows_leg(swap, pay, PAY_LEG, pay_freq)
        df_rec = self._generate_swap_cashflows_leg(swap, rec, REC_LEG, rec_freq)
        df_cashflows = pd.concat([df_pay, df_rec])

        return df_cashflows

    def generate_swap_contract(
        self,
        buysell: str,
        tenor: int,
        zerocurve: Zerocurve,
        amount: float = 100000,
        base_rate: float = 0,
        margin: float = MARGIN,
    ):
        """Generate a Interest Rate Swap contract"""
        CONTRACT_NO_OFFSET = 10000
        buysell = buysell.lower()
        contract_no = len(self.df_swaps) + 1 + CONTRACT_NO_OFFSET
        start_date = self.pos_date + DAYS_OFFSET
        if buysell == "buy":
            pay = "fixed"
            pay_freq = "annual"
            rec = "float"
            rec_freq = "semi annual"
        else:
            pay = "float"
            pay_freq = "semi annual"
            rec = "fixed"
            rec_freq = "annual"
            margin = -abs(margin)
        end_date = start_date + pd.DateOffset(months=tenor)
        if end_date.weekday() >= 5:
            end_date = end_date + BDay(1)
            end_date = end_date.to_pydatetime()
        swap = pd.DataFrame(
            [
                {
                    "contract_no": contract_no,
                    "pay": pay,
                    "start_date": start_date,
                    "pay_freq": pay_freq,
                    "rec": rec,
                    "rec_freq": rec_freq,
                    "principal": amount,
                    "tenor": tenor,
                    "base_rate": base_rate,
                    "margin": margin,
                }
            ]
        )
        df_cashflows = self._generate_swap_cashflows_(swap, zerocurve)
        self.df_swaps = pd.concat([self.df_swaps, swap])
        logger.info(
            f"Added {len(swap)} swap contracts to our portfolio. {len(self.df_swaps)} swaps in total."
        )
        # self.df_swaps.to_excel("df_swaps.xlsx")
        self.df_cashflows = pd.concat([self.df_cashflows, df_cashflows])

    def clear_swap_contracts(self):
        if len(self.df_cashflows.index):
            self.df_cashflows = self.df_cashflows[self.df_cashflows["type"] != "swap"]
        self.df_swaps.drop(self.df_swaps.index, inplace=True)

    def fixing_interest_rate_swaps(self, zerocurve: Zerocurve):
        """Fixing floating rate contracts"""
        start_date = self.pos_date
        df = self.df_cashflows
        df_swaps = self.df_swaps
        df_cflows = df[
            (df["type"] == "swap")
            & (df["rate_type"] == "float")
            & (df["rateset_dt"] <= start_date)
            & (df["cashflow"].isna())
        ]
        # Merge contracts and floating cashflows to calculate interest
        df_merged = df_swaps.merge(df_cflows, on="contract_no", how="inner")
        forward_curve = zerocurve.interpolate(start_date)
        df_merged = df_merged.merge(
            forward_curve, left_on="value_dt", right_on=forward_curve.index, how="left"
        )
        df_merged["cashflow"] = round(
            (df_merged["principal"] * df_merged["rate"] / 100 * df_merged["leg"])
            / FLOAT_FREQ_PER_YEAR,
            0,
        )
        # Update the cashflows
        df.loc[
            (df["type"] == "swap")
            & (df["rate_type"] == "float")
            & (df["rateset_dt"] <= start_date)
            & (df["cashflow"].isna()),
            "cashflow",
        ] = df_merged["cashflow"]

    def generate_nonmaturing_deposits(
        self, principal: float = 1000000, core: float = 0.4, maturity: int = 54
    ) -> pd.DataFrame:
        """Generate Non Maturing Deposits (e.g. bank and saving accounts )"""
        self.nmd_pricipal = principal
        self.nmd_core = core
        self.nmd_maturity = maturity
        noncore = 1 - core
        cashflow = []
        cashflow.append((1, round(-principal * noncore)))
        cashflow.append((2, round(-principal * core)))
        df_cashflows = pd.DataFrame(cashflow, columns=["period", "cashflow"])
        df_cashflows["value_dt"] = [
            # self.pos_date,
            self.pos_date + BDay(1),
            self.pos_date + DateOffset(months=maturity),
        ]
        df_cashflows["type"] = "deposits"
        df_cashflows["rate_type"] = "fixed"
        df_cashflows["contract_no"] = 0
        self.df_cashflows = pd.concat([self.df_cashflows, df_cashflows])
        return df_cashflows

    def step_nmd(self):
        """Roll the bank accounts and customer accounts 1 timestep forward"""
        self.df_cashflows = self.df_cashflows[self.df_cashflows["type"] != "deposits"]
        self.generate_nonmaturing_deposits(
            self.nmd_pricipal, self.nmd_core, self.nmd_maturity
        )

    def generate_funding(
        self, principal: float = 1000000, rate: float = 1, maturity: int = 120
    ) -> pd.DataFrame:
        """Generate Funding from capital market  )"""

        r = rate / 100
        interest_paid = round(-principal * r, 0)
        periods = int(round(maturity / 12, 0))
        principal_payment = 0
        cashflow = []
        for i in range(1, periods + 1):
            if i == periods:
                principal_payment = -round(principal)
            cashflow.append((i, principal_payment + interest_paid))
        df_cashflows = pd.DataFrame(cashflow, columns=["period", "cashflow"])
        df_cashflows["value_dt"] = self._date_schedule_(
            self.pos_date, len(df_cashflows), offset="years"
        )
        df_cashflows["type"] = "Funding"
        df_cashflows["rate_type"] = "fixed"
        df_cashflows["contract_no"] = 0
        self.df_cashflows = pd.concat([self.df_cashflows, df_cashflows])
        logger.info(f"Added {principal} amount in capital market funding to our model.")
        return df_cashflows

    def reset(self):
        self.pos_date = self.origin_pos_date

    def calculate_npv(self, zerocurve: Zerocurve, df_cashflows, pos_date: datetime):
        """Calculate the PV of a series of cashflows given the zero curve"""
        if pos_date == None:
            pos_date = self.pos_date
        df_forward = zerocurve.interpolate(pos_date)
        if df_forward.empty:
            logger.error("Zerocurve not found for date {pos_date}")
            return 0
        if df_cashflows.empty:
            logger.warning("No cashflows setup for model.")
            return 0
        df_c = df_cashflows[
            df_cashflows["value_dt"] > pos_date
        ]  # only include future cashflows
        if df_c.empty:
            logger.warning("No future cashflows at {pos_date}")
            return 0
        df_npv = df_c.merge(
            df_forward, left_on="value_dt", right_on=df_forward.index, how="left"
        )
        df_npv["rate"] = df_npv["rate"].ffill()

        # Something strange here with pandas. It still gives me this warning
        warnings.filterwarnings("ignore", "In a future version")

        # project future cashflows for floating leg swaps
        mask = (
            (df_npv["type"] == "swap")
            & (df_npv["rate_type"] == "float")
            & (df_npv["cashflow"].isna())
        )

        if "principal" in df_npv.columns:
            df_npv.loc[mask, "cashflow"] = (
                (
                    df_npv.loc[mask, "principal"]
                    * df_npv.loc[mask, "rate"]
                    / 100
                    * df_npv.loc[mask, "leg"]
                )
                / FLOAT_FREQ_PER_YEAR
            ).round(0)

        # df_npv.loc[mask, "cashflow"] = round(
        #     (
        #         df_npv.loc[mask, "principal"]
        #         * df_npv.loc[mask, "rate"]
        #         / 100
        #         * df_npv.loc[mask, "leg"]
        #     )
        #     / FLOAT_FREQ_PER_YEAR,
        #     0,
        # )

        df_npv["pos_dt"] = pos_date
        df_npv["year_frac"] = round(
            (df_npv["value_dt"] - df_npv["pos_dt"]) / timedelta(365, 0, 0, 0), 5
        )
        df_npv["df"] = 1 / (1 + df_npv["rate"] / 100) ** df_npv["year_frac"]
        # discount cashflows to pos_date
        df_npv["npv"] = round(df_npv["cashflow"] * df_npv["df"], 2)

        return df_npv

    # def calculate_npv(self, zerocurve: Zerocurve):
    #     """Calculate the NPV of the cashflows given the zero curve"""
    #     pos_date = self.pos_date
    #     df_forward = zerocurve.interpolate(pos_date)
    #     if df_forward.empty:
    #         logger.error("Zerocurve not found for date {pos_date}")
    #         return 0
    #     df = self.df_cashflows
    #     if df.empty:
    #         logger.warning("No cashflows setup for model.")
    #         return 0
    #     df_c = df[df["value_dt"] > self.pos_date]  # only include future cashflows
    #     if df_c.empty:
    #         logger.warning("No future cashflows at {pos_date}")
    #         return 0
    #     df_npv = df_c.merge(
    #         df_forward, left_on="value_dt", right_on=df_forward.index, how="left"
    #     )
    #     df_npv["rate"] = df_npv["rate"].ffill()
    #     # project future cashflows for floating leg swaps
    #     df_swaps = self.df_swaps
    #     if len(df_swaps.index):
    #         df_cflows = df_npv[
    #             (df_npv["type"] == "swap")
    #             & (df_npv["rate_type"] == "float")
    #             & (df_npv["cashflow"].isna())
    #         ]
    #         df_float = df_swaps.merge(df_cflows, on="contract_no", how="inner")
    #         df_float["cashflow"] = round(
    #             (df_float["principal"] * df_float["rate"] / 100 * df_float["leg"])
    #             / FLOAT_FREQ_PER_YEAR,
    #             0,
    #         )
    #         df_float.index = df_cflows.index
    #         df_npv.update(
    #             df_float["cashflow"].where(
    #                 (df_npv["type"] == "swap")
    #                 & (df_npv["rate_type"] == "float")
    #                 & (df_npv["cashflow"].isna())
    #             )
    #         )
    #     df_npv["pos_dt"] = pos_date
    #     df_npv["year_frac"] = round(
    #         (df_npv["value_dt"] - df_npv["pos_dt"]) / timedelta(365, 0, 0, 0), 5
    #     )
    #     df_npv["df"] = 1 / (1 + df_npv["rate"] / 100) ** df_npv["year_frac"]
    #     # discount cashflows to pos_date
    #     df_npv["npv"] = round(df_npv["cashflow"] * df_npv["df"], 2)

    #     df_npv.to_excel("df_npv.xlsx")

    #     return round(df_npv["npv"].sum(), 2)

    def calculate_nii(self, zerocurve: Zerocurve):
        # substraction of the interst expense - interest income
        # To calculate this correctly I need to extend the cashflow model
        # We need to seperate principal payments and interest payments
        pass

    def calculate_risk(
        self, zerocurve: Zerocurve, shock: int = 1, direction: str = "parallel"
    ):
        """Calculate a parallel shift, short shift or long shift in the zero curve"""
        pos_date = self.pos_date
        df_date = zerocurve.df.loc[pos_date].copy()
        df_date = df_date.set_index("value_dt")
        direction = direction.lower()
        if direction == "parallel":
            df_date["shock"] = shock / 100
        elif direction == "short":
            df_date["shock"] = 0
            df_date.loc[df_date["tenor"] <= 18, "shock"] = shock / 100
        elif direction == "long":
            df_date["shock"] = 0
            df_date.loc[df_date["tenor"] > 18, "shock"] = shock / 100
        else:
            logger.error(f"Unknown value for parameter {direction}.")
            df_date["shock"] = 0
        df_date["shock"] += df_date["rate"]
        df_forward = df_date[["shock", "rate"]].resample("D").mean()
        df_forward["shock"] = df_forward["shock"].interpolate()
        df_forward["rate"] = df_forward["rate"].interpolate()

        df = self.df_cashflows
        df_c = df[df["value_dt"] > self.pos_date]  # Only include future cashflows
        df_returns = df_c.merge(
            df_forward, left_on="value_dt", right_on=df_forward.index
        )
        df_returns["pos_dt"] = pos_date
        df_returns["year_frac"] = round(
            (df_returns["value_dt"] - df_returns["pos_dt"]) / timedelta(365, 0, 0, 0), 5
        )
        df_returns["df_npv"] = (
            1 / (1 + df_returns["rate"] / 100) ** df_returns["year_frac"]
        )
        df_returns["df_shock"] = (
            1 / (1 + df_returns["shock"] / 100) ** df_returns["year_frac"]
        )
        df_returns["npv"] = round(df_returns["cashflow"] * df_returns["df_npv"], 2)
        df_returns["npv_shock"] = round(
            df_returns["cashflow"] * df_returns["df_shock"], 2
        )
        return sum(df_returns["npv_shock"] - df_returns["npv"])

    def calculate_bpv(self, zerocurve: Zerocurve, shock: int = 1) -> pd.DataFrame:
        """Calculate BPV profile, applying the shock to each tenor and calculate the impact"""

        pos_date = self.pos_date
        shock = 1 / 100
        # Current zero curve
        df_zerocurve_date = zerocurve.df.loc[pos_date]

        # Bin cashflows per tenor
        bins = [
            pos_date + pd.offsets.DateOffset(months=item)
            for item in zerocurve.df.loc[pos_date].tenor.to_list()
        ]
        if self.df_cashflows["value_dt"].max() > bins[-1]:
            bins[-1] = self.df_cashflows["value_dt"].max()
        cats = zerocurve.df.loc[pos_date].tenor.to_list()[
            1:
        ]  # list(range(1, len(bins)))
        df = self.df_cashflows
        df["tenor"] = pd.cut(df["value_dt"], bins, labels=cats, right=False)
        df = df[df["value_dt"] > pos_date]
        df = df[["tenor", "cashflow"]].groupby("tenor").sum("cashflow")

        # Cashflows to numpy
        cashflow = df["cashflow"].to_numpy().reshape(-1, 1)
        # add row for tenor = 0, cashflows at T0 are not included in the cashflow list
        cashflow = np.r_[np.zeros((1, 1)), cashflow]
        # Reshape tenor to year fraction
        t = (df_zerocurve_date["tenor"] / 12).to_numpy().reshape(-1, 1)
        # Zerocurve to numpy
        rates = df_zerocurve_date["rate"].to_numpy()
        # shock rates up and down per tenor point and add as seperate columns
        shock_range = [shock]
        shocks = np.zeros((len(rates), len(rates) * len(shock_range)))
        for s in shock_range:
            for i in range(len(rates)):
                shocks[:, i + shock_range.index(s) * len(rates)] = rates
                shocks[i, i + shock_range.index(s) * len(rates)] = rates[i] + s
        # Reshape rates to concat with shocks
        rates = rates.reshape(len(rates), 1)
        # add shocks to rates
        rates = np.concatenate((rates, shocks), axis=1)
        # calculate discount factor for all rates (including shocked rates)
        discount_factor = (1 / (1 + rates / 100)) ** t
        # calculate the npv for all cashflows under all scenarios
        npv = discount_factor * cashflow
        positive = npv[:, 1 : len(rates) + 1]
        # negative = npv[:, len(rates) + 1 :]
        result = np.sum(
            np.round(positive - npv[:, 0].reshape(-1, 1), 0),
            axis=0,
        ).reshape(
            -1, 1
        )  # np.minimum(positive, negative)
        df_result = pd.DataFrame(result)
        df_result["tenor"] = df_zerocurve_date["tenor"].to_list()
        df_result.set_index("tenor", inplace=True)
        df_result.columns = ["dv01"]
        return df_result

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
        self.step_nmd()


def get_freq_per_year(freq: str) -> int:
    "map freq to freq per year"
    if freq == "annual":
        freq_per_year = 1
    elif freq == "semi annual":
        freq_per_year = 2
    else:
        freq_per_year = 12
    return freq_per_year
