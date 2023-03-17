# test dataset

import pytest

from src.data.interest import Interest

from pandas.tseries.offsets import DateOffset
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
from datetime import datetime


def test_interest_read_data():
    interest = Interest()
    interest.read_data()
    assert len(interest.df) > 100


def test_interest_set_period():
    interest = Interest()
    end = datetime.now().date()
    start = end - DateOffset(months=1)
    start = start.to_pydatetime().date()
    interest.set_period(start, end)
    assert interest.get_period() == (start, end)


def test_load_rates():
    i = Interest()
    i.read_data()
    i.save_data()

    i2 = Interest()
    i2.load_data()
    assert (
        i.df.reset_index(drop=True).all().all()
        == i2.df.reset_index(drop=True).all().all()
    )
