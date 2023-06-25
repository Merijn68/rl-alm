import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BasicEnv2(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Discrete(2)
        self.reward = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self, action):
        state = 1

        if action == 2:
            reward = 1
        else:
            reward = -1
        self.reward = reward
        truncated = False
        terminated = True
        info = {}
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        state = 0
        self.options = options
        self.seed = seed
        info = {}
        return state, info

    def render(self):
        # Implement viz
        if self.render_mode == "human":
            self.l.set_data(self.steps, self.states)
            self.writer.grab_frame()

    def close(self):
        return super().close()
