import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Assuming you have a DataFrame named 'mortgage_data' with columns: tenor, start_date, maturity, principal, interest

# Sample Data (replace this with your actual DataFrame)
data = {
    "tenor": [30, 20, 30, 10, 20],
    "start_date": [
        "2023-01-01",
        "2022-02-15",
        "2023-03-10",
        "2022-06-20",
        "2022-11-05",
    ],
    "maturity": ["2038-01-01", "2052-02-15", "2033-03-10", "2042-06-20", "2037-11-05"],
    "principal": [200000, 300000, 150000, 250000, 180000],
    "interest": [0.04, 0.035, 0.03, 0.045, 0.038],
}

mortgage_data = pd.DataFrame(data)
mortgage_data["start_date"] = pd.to_datetime(mortgage_data["start_date"])
mortgage_data["maturity"] = pd.to_datetime(mortgage_data["maturity"])

# Calculate the month of the latest data
latest_month = mortgage_data["start_date"].max().month

# Filter data for the latest month
latest_month_data = mortgage_data[
    (mortgage_data["start_date"].dt.month == latest_month)
]

# Plotting
plt.figure(figsize=(10, 6))

unique_tenors = mortgage_data["tenor"].unique()

# Plotting all mortgages
for tenor in unique_tenors:
    tenor_data = mortgage_data[mortgage_data["tenor"] == tenor]
    plt.bar(tenor, tenor_data["principal"].count(), label=f"{tenor} Years", alpha=0.5)

# Plotting new mortgages on top
for tenor in unique_tenors:
    latest_month_tenor_data = latest_month_data[latest_month_data["tenor"] == tenor]
    if latest_month_tenor_data.empty:
        continue
    plt.bar(
        tenor,
        latest_month_tenor_data["principal"].count(),
        color="red",
        label=f"New {tenor} Years",
        alpha=0.7,
        bottom=tenor_data["principal"].count(),
    )

plt.xlabel("Mortgage Tenor (Years)")
plt.ylabel("Total Principal")
plt.title("Mortgages by Tenor")
plt.legend()
plt.xticks(unique_tenors)
plt.show()
