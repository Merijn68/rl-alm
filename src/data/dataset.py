import requests
import pandas as pd
import io
from loguru import logger
from datetime import datetime, timedelta, date
from pathlib import Path
from src.data.definitions import DATA_RAW


class ECBData:
    def __init__(self):
        """Initialize ECB data object"""
        self._url_ = "https://sdw-wsrest.ecb.europa.eu/service/data/"
        # Start date is set to include swap and bank rates data
        format = "%Y-%m-%d"
        self._start_date_ = datetime.strptime("2010-06-01", format).date()
        today = date.today()
        last_month = today.replace(day=1) - timedelta(days=1)
        self._end_date_ = last_month
        # self._end_date_ = parse("2023-04-28")
        self.df = pd.DataFrame()

    def set_period(self, start_date: datetime, end_date: datetime):
        """Set start and end date for data load
        Do this before calling read_data
        """
        self._start_date_ = start_date
        self._end_date_ = end_date

    def get_period(self) -> tuple[datetime, datetime]:
        """Get start and end date from the actual data loaded"""
        if self.df.empty:
            logger.error("No data loaded")
            return None

        self._start_date_ = self.df.index[0]
        self._end_date_ = self.df.index[-1]

        return self._start_date_, self._end_date_

    def read_data(self):
        """Read data from ECB"""
        logger.info(f"reading {self.name} data from ESW.")
        url = self._url_
        key = self._key_
        parameters = {
            "startPeriod": self._start_date_.strftime("%Y-%m-%d"),
            "endPeriod": self._end_date_.strftime("%Y-%m-%d"),
        }
        try:
            response = requests.get(
                url + key, params=parameters, headers={"Accept": "text/csv"}
            )
            self.df = pd.read_csv(io.StringIO(response.text))
        except pd.errors.EmptyDataError:
            logger.error(
                f"Dataset '{self.name}' not loaded for period {self._start_date_} to {self._end_date_}"
            )
        return response

    def load_data(self):
        """Load data from file"""
        logger.info(f"loading {self.name} data from file.")
        data = Path(DATA_RAW, f"{self.name}.csv")
        if data.exists():
            self.df = pd.read_csv(data)
        else:
            logger.error(
                "Data not found. Use read_data to initalize data load from ECB"
            )

    def save_data(self):
        """Save data to file"""
        logger.info(f"saving {self.name} data to file.")
        self.df.to_csv(Path(DATA_RAW, f"{self.name}.csv"))
