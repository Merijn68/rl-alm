# A gym model wrapper for the bank model
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from gymnasium import Env
from src.models.action_space3 import ActionSpace
from src.models.observation_space3 import ObservationSpace
from src.models.bank_model3 import Bankmodel3
from src.visualization import visualize
from src.data.definitions import FIGURES_PATH

EPISODE_LENGTH = 60


class BankEnv3(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode="human"):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self.bankmodel = Bankmodel3()

        # Observation space
        self.observation_space = ObservationSpace()
        self.action_space = ActionSpace(self.bankmodel.num_actions)

        # Set episode length
        self.episode_length = EPISODE_LENGTH
        self.timestep = 0

        # Initialize scaling variables for rates and cashflows
        self.min_cashflow = 0
        self.max_cashflow = 0
        self.min_interest_rate = 0.1
        self.max_interest_rate = 0
        self.min_zero_rate = 0
        self.max_zero_rate = 0

        self.episode_rewards = []
        self.total_reward = 0
        self.episode_nii = []
        self.total_nii = 0
        self.episode_risk_penalty = []
        self.total_risk_penalty = 0
        self.episode_liquidity_penalty = []
        self.total_liquidity_penalty = 0

        self.state = self.get_state()

        # Initialize variable for mpeg writer
        self.writer = None
        self.l = None
        self.figure = None

    def step(self, action: ActionSpace):
        # Apply action
        if not (self.action_space.contains(action)):
            print(action)
            raise ValueError("Action not in action space")
        allocation = self.action_space.normalize_allocations(action)

        self.bankmodel.step(allocation, self.timestep)
        self.state = self.get_state()
        reward, nii, risk_penalty, liquidity_penalty = self.bankmodel.get_reward()
        self.total_reward += reward
        self.total_nii += nii
        self.total_risk_penalty += risk_penalty
        self.total_liquidity_penalty += liquidity_penalty

        # Reduce episode length by 1 second
        self.timestep += 1
        self.episode_length -= 1
        truncated = False
        # Check if episode is done
        if self.episode_length <= 0:
            # update statistics
            self.episode_rewards.append(self.total_reward)
            self.episode_nii.append(self.total_nii)
            self.episode_risk_penalty.append(self.total_risk_penalty)
            self.episode_liquidity_penalty.append(self.total_liquidity_penalty)
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
        """
        Plot the current situation of the bank model
        This includes projected cashflows, mortgages, funding, interest rates and zero rates
        """

        mortgages = self.bankmodel.sa_mortgages
        filter = mortgages["maturity_date"] > self.bankmodel.pos_date
        mortgages = mortgages[filter]

        funding = self.bankmodel.sa_funding
        filter = funding["maturity_date"] > self.bankmodel.pos_date
        funding = funding[filter]

        interest_rates = self.bankmodel.hullwhite.get_simulated_interest_rates(
            self.timestep - 1
        )
        zero_rates = self.bankmodel.hullwhite.get_simulated_zero_rates(
            self.timestep - 1
        )
        cf_proj_cashflows = self.bankmodel.calculate_cashflows("all")
        cf_funding = self.bankmodel.calculate_cashflows("funding")
        cf_mortgages = self.bankmodel.calculate_cashflows("mortgages")

        # Plot cashflows
        visualize.situational_plot(
            self.bankmodel.pos_date,
            cf_proj_cashflows,
            cf_funding,
            cf_mortgages,
            interest_rates,
            zero_rates,
            mortgages,
            funding,
            title="Bank model",
        )

    def plot_rewards(self):
        visualize.plot_rewards(
            self.episode_rewards,
            self.episode_nii,
            self.episode_risk_penalty,
            self.episode_liquidity_penalty,
        )

    def get_state(self):
        """the obervable state of the bank at each time step"""
        cf = self.bankmodel.calculate_cashflows("all")
        " Current interest rates"
        i = self.bankmodel.hullwhite.get_simulated_interest_rates(self.timestep - 1)
        z = self.bankmodel.hullwhite.get_simulated_zero_rates(self.timestep - 1)

        self.min_cashflow = min(self.min_cashflow, min(cf["cashflow"]))
        self.max_cashflow = max(self.max_cashflow, max(cf["cashflow"]))
        self.min_interest_rate = min(self.min_interest_rate, min(i))
        self.max_interest_rate = max(self.max_interest_rate, max(i))
        self.min_zero_rate = min(self.min_zero_rate, min(z))
        self.max_zero_rate = max(self.max_zero_rate, max(z))

        scaled_cf = (cf["cashflow"] - self.min_cashflow) / (
            (self.max_cashflow - self.min_cashflow) + 1e-6
        )
        scaled_i = (i - self.min_interest_rate) / (
            (self.max_interest_rate - self.min_interest_rate) + 1e-6
        )
        scaled_z = (z - self.min_zero_rate) / (
            (self.max_zero_rate - self.min_zero_rate) + 1e-6
        )

        state = np.concatenate((scaled_cf[:, np.newaxis], scaled_z, scaled_i)).reshape(
            self.observation_space.shape
        )
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
        self.state = self.get_state()

        # Reset the episode length
        self.episode_length = EPISODE_LENGTH
        self.timestep = 0

        # Reset the statistics
        self.total_reward = 0
        self.total_nii = 0
        self.total_risk_penalty = 0
        self.total_liquidity_penalty = 0

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
