# A gym model wrapper for the bank model

from gymnasium import Env
from src.models.action_space import ActionSpace
from src.models.observation_space import ObservationSpace
from src.models.bank_model import Bankmodel

EPISODE_LENGTH = 252


class BankEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, bm: Bankmodel, render_mode="human"):
        super().__init__()

        # Observation space
        self.observation_space = ObservationSpace(
            zerocurve=bm.zerocurve, cashflows=bm.df_cashflows
        )
        self.action_space = ActionSpace()

        # Set bank model
        self.bankmodel = bm

        # Set episode length
        self.episode_length = EPISODE_LENGTH
        self.timestep = 0

    def step(self, action):
        # Apply action
        self.bankmodel.apply_action(*self.action_space.translate_action(action))
        self.state = self.observation_space.get_observation(
            self.bankmodel.zerocurve,
            self.bankmodel.df_cashflows,
            self.bankmodel.pos_date,
        )

        # Reduce episode length by 1 second
        self.timestep += 1
        self.episode_length -= 1

        # Calculate risk and reward
        reward = self.bankmodel.get_reward()

        truncated = False
        # Check if episode is done
        if self.episode_length <= 0:
            terminated = True
        else:
            terminated = False
        # print(self.timestep, self.bankmodel.pos_date, reward, terminated)

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Implement viz
        pass

    def reset(self, seed=None):
        # Reset shower temperature
        if seed is not None:
            self.seed = seed
            self.bankmodel.set_random_seed(seed)

        self.bankmodel.reset()
        self.state = self.observation_space.get_observation(
            self.bankmodel.zerocurve,
            self.bankmodel.df_cashflows,
            self.bankmodel.pos_date,
        )
        # Reset shower time
        self.episode_length = EPISODE_LENGTH
        info = {}
        return self.state, info
