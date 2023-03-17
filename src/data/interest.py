import pandas as pd
from src.visualization import visualize
from src.data import dataset

DATAFLOW = "MIR"
FREQ = "M"
REF_AREA = "NL"
SECTOR = "B"
BALANCE_ITEM = "A2CC"
MATURITY = "F+I+O+P"
DATA_TYPE = "R"
AMOUNT_CAT = "A"
COUNTERPARTY_SECTOR = "2250"
CURRENCY = "EUR"
COVERAGE = "N"

class Interest(dataset.ECBData):
    """Read interest data from ECB Statistical Data Warehouse"""

    def __init__(self):
        super().__init__()
        self.name = "interest"
        
        self._key_ = (
            f"{DATAFLOW}/{FREQ}.{REF_AREA}.{SECTOR}.{BALANCE_ITEM}."
            f"{MATURITY}.{DATA_TYPE}.{AMOUNT_CAT}."
            f"{COUNTERPARTY_SECTOR}.{CURRENCY}.{COVERAGE}"
        )

    def read_data(self):
        response = super().read_data()
        df = self.df
        if df.empty:
            return response

        df["fixed_period"] = df["KEY"].map(
            {
                "MIR.M.NL.B.A2CC.F.R.A.2250.EUR.N": "<= 1 year",
                "MIR.M.NL.B.A2CC.I.R.A.2250.EUR.N": "1>5 years",
                "MIR.M.NL.B.A2CC.O.R.A.2250.EUR.N": "5>10 years",
                "MIR.M.NL.B.A2CC.P.R.A.2250.EUR.N": ">10 years",
            }
        )
        df = df.filter(["TIME_PERIOD", "fixed_period", "OBS_VALUE"], axis=1)
        df.columns = ["period", "fixed_period", "interest"]
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period")
        df.sort_values(["period", "fixed_period"], inplace=True)
        self.df = df
        return response

    def lineplot(self):
        visualize.lineplot(
            self.df,
            x=self.df.index,
            y="interest",
            x_label="Period",
            y_label="Interest %",
            hue="fixed_period",
            title="Loans to households for house purchase with collateral (new business) - Netherlands",
        )

    def load_data(self):
        super().load_data()
        df = self.df
        df["period"] = pd.to_datetime(df["period"])
        df.set_index("period", inplace=True)
        df.sort_values(["period", "fixed_period"], inplace=True)
        self.df = df