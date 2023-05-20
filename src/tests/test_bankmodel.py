import pytest
import pandas as pd
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
from pathlib import Path

from src.data.interest import Interest
from src.data.zerocurve import Zerocurve
from src.models.bank_model import Bankmodel
from src.data.definitions import DATA_DEBUG, DEBUG_MODE


def test_bankmodel_generate_morgage_contracts():

    pos_date = parse("28-feb-2023")
    bankmodel = Bankmodel(pos_date)

    interest = Interest()
    interest.read_data()

    bankmodel.generate_mortgage_contracts(n=10, df_i=interest.df, amount=100000)

    assert len(bankmodel.df_mortgages == 10)


def main():

    pd.options.display.float_format = "{:.2f}".format

    interest = Interest()
    interest.read_data()
    zerocurve = Zerocurve()
    zerocurve.read_data()
    pos_date = zerocurve.df.index[-1] - BDay(2)
    print(f"pos_date = {pos_date}")
    bankmodel = Bankmodel(pos_date)
    result = bankmodel.calculate_npv(
        zerocurve, bankmodel.df_cashflows, bankmodel.pos_date
    )

    print("npv = ", result)
    bankmodel.generate_mortgage_contracts(n=100, df_i=interest.df, amount=1000000)
    bankmodel.generate_swap_contract("sell", 24, zerocurve, amount=250000000)
    result = bankmodel.calculate_npv(
        zerocurve, bankmodel.df_cashflows, bankmodel.pos_date
    )
    bpv = bankmodel.calculate_bpv(zerocurve)
    print("bpv = ", bpv)
    result = bankmodel.calculate_npv(
        zerocurve, bankmodel.df_cashflows, bankmodel.pos_date
    )
    if DEBUG_MODE:
        result.to_excel(Path(DATA_DEBUG, "npv.xlsx"))
    print("npv = ", sum(result["npv"]))
    for i in range(252):
        bankmodel.step(1)
        zerocurve.step(1)
        bankmodel.fixing_interest_rate_swaps(zerocurve)
    result = bankmodel.calculate_nii(zerocurve, 1)
    if DEBUG_MODE:
        bankmodel.df_cashflows.to_excel(Path(DATA_DEBUG, "df_cashflows.xlsx"))

    print(zerocurve.df.tail(20))
    print(bankmodel.df_cashflows)

    print("nii = ", result)


if __name__ == "__main__":
    main()
