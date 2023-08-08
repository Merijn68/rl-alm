import pytest
from pandas.tseries.offsets import DateOffset
from dateutil.parser import parse
from pandas.tseries.offsets import BDay
from datetime import datetime
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_DIR))

from src.data.interest import Interest


def test_interest_read_data():
    interest = Interest()
    interest.read_data()
    assert len(interest.df) > 100


def test_interest_load_data():
    interest = Interest()
    interest.load_data()
    assert len(interest.df) > 100


def main():
    interest = Interest()
    interest.read_data()
    interest.lineplot()
    print(interest.df.tail())


if __name__ == "__main__":
    main()
