from src.models.observation_space import ObservationSpace
from src.models.action_space import ActionSpace
from gymnasium import Env
from gymnasium import spaces
import numpy as np
import gymnasium as gym
from src.data.zerocurve import Zerocurve
from src.data.interest import Interest
from src.models.bank_model import Bankmodel
from pandas.tseries.offsets import BDay

import gym
from gym import spaces
import numpy as np


def main():
    zerocurve = Zerocurve()
    zerocurve.read_data()
    interest = Interest()
    interest.read_data()

    pos_date = zerocurve.df.index[-1] - BDay(2)
    bankmodel = Bankmodel(pos_date, zerocurve)
    bankmodel.generate_mortgage_contracts(100, interest.df, 1000000)
    bankmodel.generate_swap_contract("sell", 24, amount=250000000)
    bankmodel.fixing_interest_rate_swaps()
    bankmodel.step(1)

    observation_space = ObservationSpace(zerocurve, bankmodel.df_cashflows, 5)

    # Sample an observation from the space
    observation = observation_space.sample()

    # Print the details of the observation space
    print("observation space = ", observation_space)
    print("observation = ", observation)

    action_space = ActionSpace()
    action = action_space.sample()
    action = action_space.translate_action(action)
    bankmodel.apply_action(action)
    print("action space = ", action_space)
    print("action = ", action)


if __name__ == "__main__":
    main()
