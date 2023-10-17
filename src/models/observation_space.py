import gymnasium as gym
import numpy as np

MIN_CASHFLOW = -1_000_000
MAX_CASHFLOW = +1_000_000
MAX_LIQUIDITY = 1_000_000
NUMBER_OF_INDICATORS = (
    52  # 31 Years + 15 Swap rates + 4 Bank rates + 1 spread + 1 liquidity
)
MAX_SWAP_RATE = 10
MAX_BANK_RATE = 10
MAX_FEATURE = 10


class ObservationSpace(gym.spaces.Dict):
    def __init__(self):
        liquidity_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        cashflows_space = gym.spaces.Box(low=-1, high=1, shape=(31,), dtype=float)
        swap_rates_space = gym.spaces.Box(low=-1, high=1, shape=(15,), dtype=float)
        bank_rates_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
        features_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)

        # Define scaling factors (you should define these constants)
        self.liquidity_scaling_factor = MAX_LIQUIDITY
        self.cashflow_scaling_factor = MAX_CASHFLOW
        self.swap_rates_scaling_factor = MAX_SWAP_RATE
        self.bank_rates_scaling_factor = MAX_BANK_RATE
        self.features_scaling_factor = MAX_FEATURE

        super(ObservationSpace, self).__init__(
            liquidity=liquidity_space,
            cashflows=cashflows_space,
            swap_rates=swap_rates_space,
            bank_rates=bank_rates_space,
            features=features_space,
        )

    def normalize_observation(self, obs):
        """Function to normalize observations"""
        return {
            "liquidity": obs["liquidity"] / self.liquidity_scaling_factor,
            "cashflows": obs["cashflows"] / self.cashflow_scaling_factor,
            "swap_rates": obs["swap_rates"] / self.swap_rates_scaling_factor,
            "bank_rates": obs["bank_rates"] / self.bank_rates_scaling_factor,
            "features": obs["features"] / self.features_scaling_factor,
        }

    def denormalize_observation(self, obs):
        """Function to denormalize observations"""
        return {
            "liquidity": obs["liquidity"] * self.liquidity_scaling_factor,
            "cashflows": obs["cashflows"] * self.cashflow_scaling_factor,
            "swap_rates": obs["swap_rates"] * self.swap_rates_scaling_factor,
            "bank_rates": obs["bank_rates"] * self.bank_rates_scaling_factor,
            "features": obs["features"] * self.features_scaling_factor,
        }


# class ObservationSpace(gym.spaces.Box):
#     def __init__(self):
#         # Define the bounds for normalization (used to sample the observation space)
#         self.min_cashflow = MIN_CASHFLOW
#         self.max_cashflow = MAX_CASHFLOW

#         super(ObservationSpace, self).__init__(
#             low=-np.inf,
#             high=np.inf,
#             shape=(NUMBER_OF_INDICATORS,),
#             dtype=np.float64,
#         )
