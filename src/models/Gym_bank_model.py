from gymnasium import Env, spaces
import numpy as np
import random

# We need to set the action_space and observation_space
# Action Space:
# Buy, Sell and Hold
# Amount to swap

# Observation space
# Last five days of yield data
tenor, rate

# future cashflows

# Reward: NII + NPV over time + Penalties for breaking limits
# most simple case - lets only check NII

# Bank environment for ALM Agent to train
# Bank environments will be made progressively more complex starting with a very simplified model
class BankEnv_01(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Actions we can take: 0 (sell), 1 do noting, 2 buy
        self.action_space = self.Box(
            low=np.array[0, 0], high=np.array[3.1], dtype=np.float16
        )

        # Observation space
        self.observation_space = self.Box(low=0, high=1, shape=(6, 6), dtype=np.float16)

        # Set start state
        self.state = 0.5 + random.randint(-0.3, 0.3)

        # Set episode length
        self.episode_length = 60
        self.timestep = 0

    def step(self, action):
        # Apply action
        self.state += action - 1
        # Reduce episode length by 1 second
        self.timestep += 1
        self.episode_length -= 1

        # Calculate risk and reward

        short_term_interest = get_interest(self.timestep, "ST")
        long_term_interest = get_interest(self.timestep, "LT")
        short_term_interest * state + long_term_interest * (1 - state)

        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        # Check if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        # Return step information

        return self.state, reward, done, info

    def render(self, mode):
        # Implement viz
        pass

    def reset(self):
        # Reset shower temperature
        self.state = 38 + random.randint(-3, 3)
        # Reset shower time
        self.shower_length = 60
        return self.state

    def get_interest(timestep, cl):
        if cl == "LT":
            interest = get_interest(timestep, "LT")
        else:
            interest = get_interest(timestep, "ST")
        end
        return interest[timestep]
