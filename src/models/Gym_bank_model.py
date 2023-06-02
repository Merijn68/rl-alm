# A gym model wrapper for the bank model

from gymnasium import Env
from models.action_space import ActionSpace
from models.observation_space import ObservationSpace
from models.bank_model import Bankmodel


class Gym_Bankmodel(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, bank_model: Bankmodel):
        super().__init__()

        # Observation space
        self.observation_space = ObservationSpace()
        self.action_space = ActionSpace(5)  # max 5 swaps

        # Set bank model
        self.bankmodel = bank_model

        # Set start state
        self.state = self.bankmodel.get_state()

        # Set episode length
        self.episode_length = 60
        self.timestep = 0

    def step(self, action: ActionSpace):
        # Apply action
        self.bankmodel.apply_action(action)
        self.state += self.bankmodel.get_state()

        # Reduce episode length by 1 second
        self.timestep += 1
        self.episode_length -= 1

        # Calculate risk and reward
        reward = self.bankmodel.get_reward(self.state)

        # Check if episode is done
        if self.episode_length <= 0:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self, mode):
        # Implement viz
        pass

    def reset(self):
        # Reset shower temperature
        self.bankmodel.reset()
        self.state = self.bankmodel.get_state()
        # Reset shower time
        self.episode_length = 60
        return self.state
