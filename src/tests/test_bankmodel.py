import pytest
from pandas.tseries.offsets import DateOffset
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
from datetime import datetime
from src.data.interest import Interest
from src.data.zerocurve import Zerocurve
from src.models.bank_model import Bankmodel
import pandas as pd


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
    bankmodel.generate_swap_contract("sell", 24, zerocurve, amount=250000000)
    result = bankmodel.calculate_npv(
        zerocurve, bankmodel.df_cashflows, bankmodel.pos_date
    )

    bpv = bankmodel.calculate_bpv(zerocurve)
    print("bpv = ", bpv)
    result = bankmodel.calculate_npv(
        zerocurve, bankmodel.df_cashflows, bankmodel.pos_date
    )
    result.to_excel("npv.xlsx")
    print("npv = ", sum(result["npv"]))


if __name__ == "__main__":
    main()
