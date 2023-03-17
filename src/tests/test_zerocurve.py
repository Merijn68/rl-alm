import pytest
from pandas.tseries.offsets import DateOffset
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
from datetime import datetime
from src.data.zerocurve import Zerocurve


def test_zerocurve_read_data():
    zero = Zerocurve()
    end = datetime.now().date()
    start = end - DateOffset(months=1)
    start = start.to_pydatetime().date()
    zero.set_period(start, end)
    zero.read_data()
    assert len(zero.df) > 100


def test_zero_set_period():
    zero = Zerocurve()
    end = datetime.now().date()
    start = end - DateOffset(months=1)
    start = start.to_pydatetime().date()
    zero.set_period(start, end)
    assert zero.get_period() == (start, end)


def main():
    zerocurve = Zerocurve()
    zerocurve.read_data()
    zerocurve.step()
    print(zerocurve.df.head())
    print(zerocurve.yield_data.head())


if __name__ == "__main__":
    main()
