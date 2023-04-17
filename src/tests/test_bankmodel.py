import pytest
from pandas.tseries.offsets import DateOffset
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
from datetime import datetime
from src.data.interest import Interest
from src.data.zerocurve import Zerocurve
from src.models.bank_model import Bankmodel


def test_bankmodel_generate_morgage_contracts():

    pos_date = parse("28-feb-2023")
    bankmodel = Bankmodel(pos_date)

    interest = Interest()
    interest.read_data()

    bankmodel.generate_mortgage_contracts(n=10, df_i=interest.df, amount=100000)

    assert len(bankmodel.df_mortgages == 10)


def main():

    interest = Interest()
    interest.read_data()
    zerocurve = Zerocurve()
    zerocurve.read_data()
    pos_date = zerocurve.df.index[-1] - BDay(2)
    print(f"pos_date = {pos_date}")
    bankmodel = Bankmodel(pos_date)

    # bankmodel.generate_mortgage_contracts(n=100, df_i=interest.df, amount=100000)
    # bankmodel.generate_nonmaturing_deposits(principal=9000000, core=0.4, maturity=54)
    # rate = zerocurve.df.loc[pos_date].query("tenor == 120")["rate"][0]
    # bankmodel.generate_funding(principal=1000000, rate=rate, maturity=120)
    result = bankmodel.calculate_npv(zerocurve)
    print("npv = ", result)

    bankmodel.generate_swap_contract("buy", 360, zerocurve, amount=50000000)
    bankmodel.generate_swap_contract("sell", 120, zerocurve, amount=50000000)

    # bankmodel.fixing_interest_rate_swaps(zerocurve)
    # for i in range(0, 10):
    #    zerocurve.step()
    #    bankmodel.step()
    bpv = bankmodel.calculate_bpv(zerocurve)
    print("bpv = ", bpv)
    npv = bankmodel.calculate_npv(zerocurve)
    print("npv = ", npv)


if __name__ == "__main__":
    main()
