# A gym model wrapper for the bank model
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from gymnasium import Env
from src.models.action_space2 import ActionSpace
from src.models.observation_space2 import ObservationSpace
from src.models.bank_model2 import Bankmodel2
from src.data.definitions import FIGURES_PATH

EPISODE_LENGTH = 12 * 10


class BankEnv2(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode="human"):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

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

    def step(self, action: ActionSpace):
        # Apply action
        if not (self.action_space.contains(action)):
            print(action)
            raise ValueError("Action not in action space")

        self.bankmodel.step(action, self.timestep)
        self.state = self._get_state(self.bankmodel.calculate_cashflows("all"))
        reward = self.bankmodel.get_reward()

        # Reduce episode length by 1 second
        self.timestep += 1
        self.episode_length -= 1
        truncated = False
        # Check if episode is done
        if self.episode_length <= 0:
            terminated = True
        else:
            terminated = False

        # Set placeholder for info
        info = {}

        return self.state, reward, terminated, truncated, info

    def render(self):
        # Implement viz
        if self.render_mode == "human":
            cf = self.bankmodel.calculate_cashflows("all")
            self.ax.cla()
            if len(cf) > 0:
                sns.barplot(x=range(len(cf)), y=cf["cashflow"], ax=self.ax)

            if self.writer is not None:
                self.writer.grab_frame()

    def plot(self):
        """Plot the cashflows of the bank model for testing"""
        cf_funding = self.bankmodel.calculate_cashflows("funding")
        cf_mortgages = self.bankmodel.calculate_cashflows("mortgages")
        cf_all = self.bankmodel.calculate_cashflows("all")
        x = np.arange(len(cf_funding))
        width = 0.23
        _, ax = plt.subplots()
        ax.bar(x - width / 2, cf_funding["cashflow"], width, label="Funding")
        ax.bar(x + width / 2, cf_mortgages["cashflow"], width, label="Mortgages")
        ax.plot(
            x, cf_all["cashflow"], color="red", linestyle="-", marker="o", label="all"
        )

        ax.set_xlabel("years")
        ax.set_ylabel("cashflow")
        ax.set_title("Future cashflows")
        ax.set_xticks(x)
        ax.legend(loc="upper left")
        plt.show()

    def _get_state(self, cf: np.ndarray):
        pos_date = self.bankmodel.pos_date
        if len(cf) == 0:
            state = np.zeros(self.observation_space.shape, dtype=np.int32)
        else:
            state = cf["cashflow"]
        if not (self.observation_space.contains(state)):
            print(state.shape)
            print(state)
            raise ValueError("State not in observation space")

        return state

    def reset(self, seed=None, options=None):
        # Reset the bank model
        if seed is not None:
            self.seed = seed

        self.bankmodel.reset()

        self.state = self._get_state(self.bankmodel.calculate_cashflows("all"))

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

        ax = self.figure.gca()
        ax.set_title("Projected cashflows")
        ax.set_ylim(-1000, +1000)
        ax.set_xlim(0, 30)
        self.ax = ax

        # This will need to be changed to match your directory.
        plt.rcParams["animation.ffmpeg_path"] = Path(
            r"C:\Program Files\ffmpeg-6.0-essentials_build\bin", r"ffmpeg.exe"
        )

        writer = FFMpegWriter(fps=5, metadata=moviedata)
        writer.setup(
            self.figure, Path(FIGURES_PATH, filename).with_suffix(".mp4"), dpi=100
        )

        self.writer = writer
