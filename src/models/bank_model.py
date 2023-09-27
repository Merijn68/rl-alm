# Simplified bank model
import numpy as np

from random import randint, randrange
from loguru import logger
from typing import Tuple

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_DIR))

from src.data.interest import Interest
from src.data.zerocurve import Zerocurve
from src.models.hullwhite import HullWhiteModel
from src.models.action_space import ActionSpace

RANGE_TENOR = [1, 5, 10, 20]  # Duration of fixed interest mortgages in years
FUNDING_TENORS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    15,
    20,
    25,
]  # Expiriment with different funding tenors
RANGE_PROBABILITIES = [
    0.08,
    0.19,
    0.23,
    0.50,
]  # Probabilities of selling mortgages with different durations
MORTGAGE_AMOUNT = 1000
BOND_AMOUNT = 1000
MORTGAGE_SIZE = 120
COLUMN_DATA = [
    ("tenor", int),
    ("start_date", "datetime64[D]"),
    ("maturity_date", "datetime64[D]"),
    ("principal", int),
    ("interest", float),
    ("period", "datetime64[Y]"),
]


class Bankmodel:
    def __init__(self) -> None:
        # Read interest data
        self.interest = Interest()
        self.interest.read_data()

        # match data periods for interest and zerocurve
        start_date, end_date = self.interest.get_period()
        self.zerocurve = Zerocurve()
        self.zerocurve.set_period(start_date, end_date)
        # read zerocurve data
        self.zerocurve.read_data()
        start_date, end_date = self.zerocurve.get_period()
        self.interest.set_period(start_date, end_date)

        print("Interest period: ", self.interest.get_period())
        print("Zerocurve period: ", self.zerocurve.get_period())

        # fit the Hull-White model to simulate interest rates
        self.hullwhite = HullWhiteModel()
        self.hullwhite.fit(self.interest, self.zerocurve)
        self.hullwhite.transform()

        # initialize the remaining fields for the bank model
        self.pos_date = end_date.to_numpy()
        self.timestep = 0
        self.liquidity = 0
        self.risk_limit = 5 * MORTGAGE_AMOUNT
        self.sa_mortgages = None
        self.sa_funding = None
        self.funding_tenors = FUNDING_TENORS
        self.num_actions = len(FUNDING_TENORS)

    def _random_date_(self, start: datetime, end: datetime) -> datetime:
        """return a random business date between start and end date"""
        delta = end - start
        date = start + timedelta(days=randrange(delta.days))
        # shift dates to next business day if closed
        if date.weekday() >= 5:
            date = date + BDay(1)
            date = date.to_pydatetime()
        return date

    def generate_mortgage_contracts(
        self,
        n: int = MORTGAGE_SIZE,
        amount: float = MORTGAGE_AMOUNT,
        probabilities=RANGE_PROBABILITIES,
    ):
        """Generate mortgage contracts for a specific period"""

        """ If we can't fund the mortgages - we can not sell more mortgages """
        if self.liquidity < n * amount:
            return

        # We use structured arrays to store the data - as dataframes are too slow
        dtypes = COLUMN_DATA
        data = np.empty(n, dtype=dtypes)
        data["tenor"] = np.random.choice(RANGE_TENOR, size=n, p=probabilities)
        data["start_date"] = np.full(n, self.pos_date)
        # delta = np.array(
        #    [timedelta(days=365 * int(y)) for y in data["tenor"]],
        #    dtype="timedelta64[D]",
        # )
        delta = np.timedelta64(365, "D") * data["tenor"]
        data["maturity_date"] = data["start_date"] + delta
        data["principal"] = np.ones(n) * amount
        data["period"] = np.array([np.datetime64(dt, "Y") for dt in data["start_date"]])
        data["interest"] = np.array(
            [
                self.get_sim_interest_rate(self.timestep, tenor)
                for tenor in data["tenor"]
            ]
        )
        self.liquidity -= n * amount
        if self.sa_mortgages is None:
            self.sa_mortgages = data
        else:
            self.sa_mortgages = np.concatenate((self.sa_mortgages, data))

    def get_sim_interest_rate(self, timestep, tenor):
        """Get interest rate for a specific start date and tenor"""
        x = RANGE_TENOR.index(tenor)
        rate = self.hullwhite.get_simulated_interest_rates(timestep)[x, 0]
        return rate

    def get_sim_zero_rate(self, timestep, tenor):
        """Get zero rate for a specific start date and tenor"""
        rates = self.hullwhite.get_simulated_zero_rates(timestep)[:, 0]
        tenor_per_month = tenor * 12

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

        # Extract the tenor values from the TENORS dictionary
        tenor_values = np.array(list(TENORS.values()))

        # Linearly interpolate the rate using numpy's interp function
        interpolated_rate = np.interp(tenor_per_month, tenor_values, rates)

        return interpolated_rate

    def buy_sell_bond(self, buy_sell, principal: float = BOND_AMOUNT, tenor: int = 5):
        """Add Capital Markets Funding"""
        dtypes = COLUMN_DATA
        data = np.empty(1, dtype=dtypes)
        data["tenor"] = tenor
        data["start_date"] = self.pos_date
        delta = np.array(
            [timedelta(days=365 * int(y)) for y in data["tenor"]],
            dtype="timedelta64[D]",
        )
        data["maturity_date"] = data["start_date"] + delta
        data["principal"] = principal * buy_sell * -1
        data["period"] = np.array([np.datetime64(dt, "Y") for dt in data["start_date"]])
        data["interest"] = np.array(
            [self.get_sim_zero_rate(self.timestep, tenor) for tenor in data["tenor"]]
        )

        # add to liquidity
        self.liquidity += principal * buy_sell

        if self.sa_funding is None:
            self.sa_funding = data
        else:
            self.sa_funding = np.concatenate((self.sa_funding, data))

    def step(self, actions=None, timestep=None) -> None:
        """Each step we fund a time bucket or not.
        We asume we can not repay the funding before maturity."""
        if timestep is not None:
            self.timestep = timestep

        from_date = self.pos_date
        self.pos_date = np.datetime64(
            self.pos_date.astype("datetime64[M]") + np.timedelta64(1, "M")
        )
        if actions is not None:
            tenor = 0
            for action in actions:
                if action == 0:
                    pass
                else:
                    self.buy_sell_bond(buy_sell=action, tenor=FUNDING_TENORS[tenor])
                tenor += 1

        # receive payments from mortgages
        if self.sa_mortgages is not None:
            self.liquidity += np.where(
                (self.sa_mortgages["maturity_date"] >= from_date)
                & (self.sa_mortgages["maturity_date"] < self.pos_date),
                self.sa_mortgages["principal"],
                0,
            ).sum()
        # pay back funding
        if self.sa_funding is not None:
            self.liquidity += np.where(
                (self.sa_funding["maturity_date"] >= from_date)
                & (self.sa_funding["maturity_date"] < self.pos_date),
                self.sa_funding["principal"],
                0,
            ).sum()

        self.generate_mortgage_contracts(
            int(
                (MORTGAGE_SIZE + randint(-MORTGAGE_SIZE * 0.1, MORTGAGE_SIZE * 0.1))
                / 12
            )
        )
        self.timestep = self.timestep + 1

    def calculate_nii(self) -> Tuple[float, float, float]:
        """Calculate the Net Interest Income per period"""

        income = 0
        funding_cost = 0

        if self.sa_mortgages is not None:
            filter = self.sa_mortgages["maturity_date"] >= self.pos_date
            mortgages = self.sa_mortgages[filter]
            income = ((mortgages["interest"] / 100) * mortgages["principal"] / 12).sum()

        if self.sa_funding is not None:
            filter = self.sa_funding["maturity_date"] >= self.pos_date
            funding = self.sa_funding[filter]
            funding_cost = (
                (funding["interest"] / 100) * funding["principal"] / 12
            ).sum()

            if funding_cost > 0:
                print("What is going on here?")
                print(f"funding_cost = {funding_cost}")
                print(self.sa_funding)

        nii = income + funding_cost
        return nii, income, funding_cost

    def calculate_cashflows(self, type: str = "all") -> np.ndarray:
        """Calculate the future principal cashflows from mortgages and funding"""

        # Create cashflow_data array
        projected_years = 31  # added 1 for the current year
        pos_date = np.datetime64(self.pos_date)
        end_date = np.datetime64(
            pos_date.astype("datetime64[Y]") + np.timedelta64(projected_years, "Y")
        )
        dtype = [("year", "datetime64[Y]"), ("cashflow", int)]
        cashflow_data = np.zeros(projected_years, dtype=dtype)
        cashflow_data["year"] = np.arange(pos_date, end_date, dtype="datetime64[Y]")

        if type not in ["mortgages", "funding", "all"]:
            raise ValueError("type must be mortgages, funding or all")
        if type == "mortgages":
            if self.sa_mortgages is None:
                return cashflow_data
            data = self.sa_mortgages
        elif type == "funding":
            if self.sa_funding is None:
                return cashflow_data
            data = self.sa_funding
        else:
            if self.sa_funding is None:
                if self.sa_mortgages is None:
                    return cashflow_data
                data = self.sa_mortgages
            else:
                if self.sa_mortgages is None:
                    data = self.sa_funding
                else:
                    data = np.concatenate((self.sa_funding, self.sa_mortgages))

        maturity_dates = np.array(data["maturity_date"], dtype="datetime64[Y]")
        principals = np.array(data["principal"], dtype=float)
        start_dates = np.array(data["start_date"], dtype="datetime64[Y]")

        # Calculate cashflows

        # receivables
        rec_mask = maturity_dates >= pos_date
        rec_value_date = maturity_dates[rec_mask]
        rec_cashflows = principals[rec_mask]
        rec_indices = np.searchsorted(
            cashflow_data["year"],
            rec_value_date.astype("datetime64[Y]"),
        )
        np.add.at(cashflow_data["cashflow"], rec_indices, rec_cashflows)

        # payments
        pay_mask = start_dates >= pos_date
        pay_value_date = start_dates[pay_mask]
        pay_cashflows = -1 * principals[pay_mask]
        pay_indices = np.searchsorted(
            cashflow_data["year"],
            pay_value_date.astype("datetime64[Y]"),
        )
        np.add.at(cashflow_data["cashflow"], pay_indices, pay_cashflows)

        return cashflow_data

    def reset(self):
        self.timestep = 0
        self.liquidity = 0  # We don't have any cash at the beginning
        _, end_date = self.interest.get_period()
        self.pos_date = end_date.to_numpy()
        self.hullwhite.transform()  # Start with new simulated rates
        self.sa_mortgages = None
        self.sa_funding = None

    def get_risk_penalty(self):
        cf = self.calculate_cashflows("all")
        risk_penalty = (
            (
                abs(
                    cf[abs(cf["cashflow"]) > self.risk_limit]["cashflow"]
                    - self.risk_limit
                ).sum()
                / self.risk_limit
            )
            * MORTGAGE_AMOUNT
            / 12
        )
        return risk_penalty

    def get_reward(self):
        nii, income, funding_cost = self.calculate_nii()
        cf = self.calculate_cashflows("all")

        # Liquidity penalty if we can not pay the projected cashflows
        if cf[0]["cashflow"] + self.liquidity < 0:
            liquidity_penalty = MORTGAGE_AMOUNT / 12
        else:
            liquidity_penalty = 0
        risk_penalty = self.get_risk_penalty()
        reward = nii - risk_penalty - liquidity_penalty
        return int(reward), int(nii), int(risk_penalty), int(liquidity_penalty)

    def draw_cashflows(self):
        c = self.calculate_cashflows()
        plt.title("Expected Cashflows")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        x = np.arange(
            self.pos_date,
            self.pos_date + np.timedelta64(31, "Y"),
            dtype="datetime64[Y]",
        )
        y = c["cashflow"]
        plt.plot(x, y, "-")
        plt.show()


def main():
    bankmodel = Bankmodel()
    bankmodel.reset()
    actionspace = ActionSpace(len(FUNDING_TENORS))

    score = 0
    for _ in range(60):
        bankmodel.step(actionspace.normalize_allocations(actionspace.sample()))
        reward, _, _, _ = bankmodel.get_reward()
        score = score + reward
    print(f"score = {score}")
    bankmodel.draw_cashflows()


if __name__ == "__main__":
    main()
