"""
    Create data sets from raw files to work with

"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from random import randrange


def generate_mortgage_cashflow(
    principal, interest_rate, years, fixed_period
) -> pd.DataFrame:
    """Generate mortgage cashflows using repricing method for lineair interest model"""

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
