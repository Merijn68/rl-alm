# test dataset

import pytest
from src.data import dataset
from pandas.tseries.offsets import DateOffset
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
from datetime import datetime


def test_interest_read_data():
    interest = dataset.Interest()
    interest.read_data()
    assert len(interest.df) > 100


def test_interest_set_period():
    interest = dataset.Interest()
    end = datetime.now().date()
    start = end - DateOffset(months=1)
    start = start.to_pydatetime().date()
    interest.set_period(start, end)
    assert interest.get_period() == (start, end)


def test_zerocurve_read_data():
    zero = dataset.Zerocurve()
    end = datetime.now().date()
    start = end - DateOffset(months=1)
    start = start.to_pydatetime().date()
    zero.set_period(start, end)
    zero.read_data()
    assert len(zero.df) > 100


def test_zero_set_period():
    zero = dataset.Zerocurve()
    end = datetime.now().date()
    start = end - DateOffset(months=1)
    start = start.to_pydatetime().date()
    zero.set_period(start, end)
    assert zero.get_period() == (start, end)


def test_load_rates():
    i = dataset.Interest()
    i.read_data()
    i.save_data()

    i2 = dataset.Interest()
    i2.load_data()
    assert (
        i.df.reset_index(drop=True).all().all()
        == i2.df.reset_index(drop=True).all().all()
    )


def main():
    i = dataset.Interest()
    response = i.read_data()
    print(response.url)
    print(i.df)

    i.save_data()

    i2 = dataset.Interest()
    i2.load_data()

    print(i2.df)
    assert (
        i.df.reset_index(drop=True).all().all()
        == i2.df.reset_index(drop=True).all().all()
    )


if __name__ == "__main__":
    main()
