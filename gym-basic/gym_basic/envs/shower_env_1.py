from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FFMpegWriter
import seaborn as sns
from src.data.definitions import FIGURES_PATH, FFMPEG_PATH
from pathlib import Path


class ShowerEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, start=None, render_mode=None, seed=None):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.start = start
        self.render_mode = render_mode
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(
            low=np.array([0]), high=np.array([1]), dtype=np.float64
        )
        # self.observation_space = Box(0, 1, shape=(1,), dtype=np.float32)

        self.reset(seed=seed)

        # Initialize variable for mpeg writer
        self.writer = None

    def step(self, action):
        self.step_count += 1
        self.states.append(self.state)
        self.actions.append(action)
        self.steps.append(self.step_count)

        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        self.state += action - 1
        # Reduce shower length by 1 second
        self.shower_length -= 1

        # Calculate reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        self.rewards.append(reward)

        # Check if shower is done
        if self.shower_length <= 0:
            terminated = True
        else:
            terminated = False

        # Apply temperature noise
        self.state += random.uniform(-1, 1)
        # Set placeholder for info
        info = {}

        # Return step information
        # return np.array([np.float32(self.state)]), reward, terminated, terminated, info
        return (
            np.array([(self.state / 100)]).astype(np.float64),
            reward,
            terminated,
            terminated,
            info,
        )

    def render(self):
        # Implement viz
        if self.render_mode == "human":
            self.l.set_data(self.steps, self.states)
            if self.writer is not None:
                self.writer.grab_frame()

    def set_render_output(self, filename: str, title: str = "Movie"):
        if self.writer is not None:
            self.writer.finish()

        moviedata = dict(title=title, artist="M. van Miltenburg")
        self.figure = plt.figure()

        # This will need to be changed to match your directory.
        plt.rcParams["animation.ffmpeg_path"] = Path(FFMPEG_PATH, r"ffmpeg.exe")
        plt.xlim(0, 60)
        plt.ylim(0, 100)
        (self.l,) = plt.plot([], [], "k-")

        # highlight the ideal temerature range
        plt.axhspan(37, 39, color="blue", alpha=0.3)

        writer = FFMpegWriter(fps=self.metadata["render_fps"], metadata=moviedata)
        outfile = Path(FIGURES_PATH, filename).with_suffix(".mp4")
        writer.setup(self.figure, outfile, dpi=100)
        self.writer = writer

    def reset(self, seed=None, options=None):
        # Reset shower temperature
        self.state = 38 + random.uniform(-5, 5)
        self.seed = seed
        self.options = options
        self.states = []
        self.rewards = []
        self.steps = []
        self.actions = []
        self.step_count = 0

        # Reset shower time
        self.shower_length = 60
        # information dic
        info = {}
        # print(f'reset the shower, current temperature is {self.state}')
        # return np.array([np.float32(self.state)]), info
        return np.array([self.state / 100]).astype(np.float64), info

    def close(self):
        if self.figure is not None:
            self.writer.finish()
        super().close()
