import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create a sample dataset (replace this with your own data)
import pandas as pd
import numpy as np


def main():
    data = {
        "period": ["2023-01", "2023-02", "2023-03", "2023-01", "2023-02", "2023-03"],
        "fixed_period": ["A", "A", "A", "B", "B", "B"],
        "interest": [2.5, 2.7, 2.6, 3.0, 3.2, 3.1],
    }

    df = pd.DataFrame(data)

    # Convert the 'period' column to datetime type
    df["period"] = pd.to_datetime(df["period"])

    # Sort the DataFrame by 'fixed_period' and 'period'
    df.sort_values(by=["fixed_period", "period"], inplace=True)

    # Pivot the DataFrame to wide format
    pivot_df = df.pivot(index="period", columns="fixed_period", values="interest")

    # Plot the line chart
    ax = pivot_df.plot(kind="line", marker="o", figsize=(10, 6))

    # Customize the plot
    ax.set_xlabel("Period")
    ax.set_ylabel("Interest")
    ax.set_title("Interest Rates Over Time")
    ax.legend(title="Fixed Period")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
