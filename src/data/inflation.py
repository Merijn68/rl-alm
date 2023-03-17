import pandas as pd
from src.visualization import visualize
from src.data import dataset

DATAFLOW = "ICP"
FREQ = "M"
REF_AREA = "NL"
ADJUSTMENT = "N"
CLASS = "000000"
PROVIDER = "4"  # ECB
VALIDATION = "ANR"

class Inflation(dataset.ECBData):
    """Read inflation data from ECB Statistical Data Warehouse"""

    def __init__(self):
        super().__init__()
        self.name = "inflation"
        
        self._key_ = (
            f"{DATAFLOW}/{FREQ}.{REF_AREA}."
            f"{ADJUSTMENT}.{CLASS}.{PROVIDER}.{VALIDATION}"
        )

    def read_data(self):
        response = super().read_data()
        df = self.df
        if df.empty:
            return response
        df = self.df.filter(["TIME_PERIOD", "OBS_VALUE"], axis=1)
        df.columns = ["period", "inflation"]
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period")
        self.df = df
        return response

    def lineplot(self):
        if self.df.empty:
            return
        visualize.lineplot(
            self.df,
            x=self.df.index,
            y="inflation",
            x_label="Period",
            y_label="Inflation %",
            title="HICP Annual rate of change Eurostat",
        )