"""

The Hull White model is an extension of the Vasicek model that allows the 
short-term interest rate to be stochastic. The model is defined by the
following stochastic differential equation:

dr(t) = [theta(t) - a * r(t)] * dt + sigma * dW(t)

where r(t) is the short-term interest rate at time t, theta(t) is the
mean-reversion level of the short-term interest rate at time t, a is the
speed of mean reversion, sigma is the volatility of the short-term interest
rate, and W(t) is a Wiener process.

"""

import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_DIR))

from src.data.interest import Interest
from src.data.zerocurve import Zerocurve
import src.visualization.visualize as visualize

SIM_HORIZON_IN_YEARS = 5  # Horizon for simulation of interest rate paths
SIM_NUM_PATHS = 1  # Number of simulated interest rate paths


class HullWhiteModel:
    def __init__(self):
        self.T = SIM_HORIZON_IN_YEARS
        self.num_steps = SIM_HORIZON_IN_YEARS * 12
        self.num_paths = SIM_NUM_PATHS
        self.num_interest_rate = 0

    def calculate(self, r0, kappa, theta, sigma, correlation_matrix):
        """Simulate interest rate paths using the Hull-White model for all correlated rates."""

        T = self.T
        num_steps = self.num_steps
        num_paths = self.num_paths

        dt = T / num_steps
        num_tenors = correlation_matrix.shape[0]

        # Initialize zero curve rates for all tenors
        rates = np.zeros((num_tenors, num_steps + 1, num_paths))
        rates[:, 0, :] = r0.reshape(num_tenors, 1)

        # Add a small positive constant (e.g., 1e-6) to the diagonal elements for regularization
        reg_corr_matrix = (
            correlation_matrix + np.eye(correlation_matrix.shape[0]) * 1e-6
        )

        for i in range(1, num_steps + 1):
            # Generate correlated random increments for all tenors and paths
            dW = np.random.normal(0, np.sqrt(dt), (num_tenors, num_paths))
            dW_corr = np.linalg.cholesky(reg_corr_matrix) @ dW

            # Update rates for all tenors and paths using the Hull-White model dynamics
            for j in range(num_tenors):
                rates[j, i, :] = (
                    rates[j, i - 1, :]
                    + kappa[j] * (theta[j] - rates[j, i - 1, :]) * dt
                    + sigma[j] * np.sqrt(dt) * dW_corr[j, :]
                )

        # Drop the initial zero rate
        rates = rates[:, 1:, :]

        return rates

    def fit(self, interest: Interest, zerocurve: Zerocurve):
        # Transform the zero curve data to a numpy array per tenor
        # TODO -> Split this so fitting the data and constructing an array is separate
        zero_df = (
            zerocurve.df[["tenor", "rate"]]
            .groupby("tenor")
            .resample("MS")
            .mean()
            .droplevel(0)
        )
        zero_df.reset_index(inplace=True)
        self.start_date = zero_df["rate_dt"].min()

        zero_rates = zero_df.pivot(
            columns="rate_dt", index="tenor", values="rate"
        ).to_numpy()

        # Convert the bank rates to a numpy array per tenor
        interest_dt = interest.df.reset_index()
        interest_rates = (
            interest_dt.pivot(columns="period", index="fixed_period", values="interest")
        ).to_numpy()

        # join zero rates and bank interest rates
        rates = np.concatenate((zero_rates, interest_rates), axis=0)

        # Shift zero rates to positive values needed for the White Hall model
        min_rate = np.min(rates)
        if min_rate < 0:
            shifted_rates = rates - min_rate + 0.01
        else:
            shifted_rates = rates + 0.01

        # Calculate the correlation matrix
        shifted_rates_transpose = np.transpose(shifted_rates)
        corr = np.corrcoef(shifted_rates_transpose, rowvar=False)

        # Calculate the simulated rates
        theta = np.mean(shifted_rates, axis=1)
        variance = np.var(shifted_rates, axis=1, ddof=1)
        kappa = (1 / theta) * np.log(theta + variance + 1)
        sigma = np.sqrt(variance)

        self.r0 = shifted_rates[:, -1]
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.corr = corr
        self.min_rate = min_rate
        self.num_interest_rates = interest_rates.shape[0]
        self.rates = rates

    def transform(self):
        simulated_rates = self.calculate(
            self.r0, self.kappa, self.theta, self.sigma, self.corr
        )

        # Shift the simulated rates back to the original values
        if self.min_rate < 0:
            self.simulated_rates = simulated_rates + self.min_rate - 0.01
        else:
            self.simulated_rates = simulated_rates - 0.01

        return self.simulated_rates

    def fit_transform(self, interest: Interest, zerocurve: Zerocurve):
        self.fit(interest, zerocurve)
        self.transform()

    def get_simulated_interest_rates(self, step):
        return self.simulated_rates[-self.num_interest_rates :, step, :]

    def get_simulated_zero_rates(self, step):
        return self.simulated_rates[: -self.num_interest_rates, step, :]

    def plot(self):
        # Plot the resulting simulated and original rate curves
        visualize.curveplot(self.rates, self.simulated_rates, self.start_date)


def main():
    interest = Interest()
    interest.read_data()
    interest.save_data()
    start_date, end_date = interest.get_period()
    zerocurve = Zerocurve()
    zerocurve.set_period(start_date, end_date)
    zerocurve.read_data()
    zerocurve.save_data()
    hullwhite = HullWhiteModel()
    hullwhite.fit(interest, zerocurve)
    hullwhite.transform()
    hullwhite.fit_transform(interest, zerocurve)
    hullwhite.plot()


if __name__ == "__main__":
    main()
