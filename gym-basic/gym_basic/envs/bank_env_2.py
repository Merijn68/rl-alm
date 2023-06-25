# A gym model wrapper for the bank model

from gymnasium import Env
from src.models.action_space2 import ActionSpace
from src.models.observation_space2 import ObservationSpace
from src.models.bank_model2 import Bankmodel2

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from src.data.definitions import FIGURES_PATH
from pathlib import Path

EPISODE_LENGTH = 252


class BankEnv2(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode="human"):
        super().__init__()

        self.bankmodel = Bankmodel2()

        # Observation space
        self.observation_space = ObservationSpace()
        self.action_space = ActionSpace()

        # Set episode length
        self.episode_length = EPISODE_LENGTH
        self.timestep = 0

        # Initialize variable for mpeg writer
        self.writer = None
        self.l = None
        self.figure = None

    def step(self, action):
        # Apply action
        self.bankmodel.step(action)
        self.state = self.bankmodel.calculate_mortgage_cashflows()

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

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Implement viz
        if self.render_mode == "human":
            future_cashflows = self.bankmodel.calculate_mortgage_cashflows()
            self.l.xdata = range(len(future_cashflows))
            self.l.ydata = future_cashflows
            plt.show()
            self.writer.grab_frame()

    def reset(self, seed=None, options=None):
        # Reset the bank model
        if seed is not None:
            self.seed = seed

        self.bankmodel.reset()
        self.state = self.bankmodel.calculate_mortgage_cashflows()

        # Reset the episode length
        self.episode_length = EPISODE_LENGTH

        info = {}
        return self.state, info

    def close(self):
        super().close()

    def set_render_output(self, filename: str, title: str = "Movie"):
        if self.writer is not None:
            self.writer.finish()

        moviedata = dict(title=title, artist="M. van Miltenburg")
        self.figure = plt.figure()
        # This will need to be changed to match your directory.
        plt.rcParams["animation.ffmpeg_path"] = Path(
            r"C:\Program Files\ffmpeg-6.0-essentials_build\bin", r"ffmpeg.exe"
        )
        # Set length of x-axis to 30 years
        (self.l,) = plt.plot([], [], "k-")

        # highlight the ideal temerature range
        plt.axhspan(37, 39, color="blue", alpha=0.3)

        writer = FFMpegWriter(fps=5, metadata=moviedata)
        outfile = Path(FIGURES_PATH, filename).with_suffix(".mp4")
        writer.setup(self.figure, outfile, dpi=100)
        self.writer = writer
