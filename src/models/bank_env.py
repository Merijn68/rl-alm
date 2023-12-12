# A gym model wrapper for the bank model
import numpy as np
import pandas as pd
import seaborn as sns
from gymnasium import Env
from src.models.action_space import ActionSpace
from src.models.observation_space import ObservationSpace
from src.models.bank_model import Bankmodel
from src.visualization import visualize
from src.data.definitions import FIGURES_PATH
from matplotlib import pyplot as plt

EPISODE_LENGTH = 60

# Change


class BankEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode="human"):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self.bankmodel = Bankmodel()

        # Observation space
        self.observation_space = ObservationSpace()
        self.action_space = ActionSpace(self.bankmodel.num_actions)

        # Set episode length
        self.episode_length = EPISODE_LENGTH
        self.timestep = 0

        self.reset_episode_statistics()

        self.episode_rewards = []
        self.episode_nii = []
        self.episode_risk_penalty = []
        self.episode_liquidity_penalty = []

        self.total_reward = 0
        self.total_nii = 0
        self.total_risk_penalty = 0
        self.total_liquidity_penalty = 0

        self.state = self.get_state()

    def step(self, action: ActionSpace):
        # Apply action
        if not (self.action_space.contains(action)):
            print(action)
            raise ValueError("Action not in action space")
        action = self.action_space.denormalize_action(action)
        self.bankmodel.step(action, self.timestep)

        self.state = self.get_state()
        (
            reward,
            nii,
            risk_penalty,
            liquidity_penalty,
        ) = self.bankmodel.get_reward()
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

    def render(self) -> np.ndarray[np.uint8]:
        """Render the current state of the bank model"""
        frame = self.get_current_frame()
        return frame

    def get_current_frame(self) -> np.ndarray[np.uint8]:
        """Get the current frame to create a video of the bank model"""
        cf = self.bankmodel.calculate_cashflows("all")

        fig, ax = plt.subplots()

        if len(cf) > 0:
            f = pd.DataFrame(self.bankmodel.sa_funding)
            f = f[f["maturity_date"] >= self.bankmodel.pos_date]
            m = pd.DataFrame(self.bankmodel.sa_mortgages)
            m = m[m["maturity_date"] >= self.bankmodel.pos_date]

            visualize.plot_frame(cf, f, m, ax=ax)

        # Convert the Matplotlib figure to an RGB array
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)  # Close the Matplotlib figure

        return frame

    def list_model(self) -> pd.DataFrame:
        """List the current situation of the bank model at each time step"""
        bankmodel = self.bankmodel

        if self.df_model is None:
            self.df_model = pd.DataFrame()

        df_funding = pd.DataFrame(bankmodel.sa_funding)
        df_mortgages = pd.DataFrame(bankmodel.sa_mortgages)

        if len(df_funding) == 0:
            total_funding = 0
        else:
            filter = df_funding["maturity_date"] >= bankmodel.pos_date
            df_fs = (
                df_funding[filter].groupby(["tenor"])["principal"].sum().reset_index()
            )
            total_funding = df_fs.principal.sum()

        if len(df_mortgages) == 0:
            total_mortgages = 0
        else:
            filter = df_mortgages["maturity_date"] >= bankmodel.pos_date
            df_m = (
                df_mortgages[filter].groupby(["tenor"])["principal"].sum().reset_index()
            )
            total_mortgages = df_m.principal.sum()

        nii, income, funding_cost = bankmodel.calculate_nii()
        reward, nii, risk_penalty, liquidity_penalty = bankmodel.get_reward()
        results = {
            "timestep": self.timestep,
            "reward": reward,
            "risk_penalty": risk_penalty,
            "liquidity_penalty": liquidity_penalty,
            "liquidity": bankmodel.liquidity,
            "nii": nii,
            "income": income,
            "funding_cost": funding_cost,
            "total_funding": total_funding,
            "total_mortgages": total_mortgages,
        }
        df_results = pd.DataFrame([results])

        self.df_model = pd.concat([self.df_model, df_results], ignore_index=True)

        return self.df_model

    def plot(self):
        """
        Plot the current situation of the bank model
        This includes projected cashflows, mortgages, funding,
        interest rates and zero rates
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
            self.timestep,
            self.bankmodel.liquidity,
            self.total_reward,
            self.total_risk_penalty,
            cf_proj_cashflows,
            cf_funding,
            cf_mortgages,
            interest_rates,
            zero_rates,
            mortgages,
            funding,
            title="Bank model",
        )

        # Add a curve plot to zoom in on the zero curve
        # self.bankmodel.hullwhite.plot()

    def plot_rewards(self):
        visualize.plot_rewards(self.episode_rewards)
        self.bankmodel.hullwhite.plot()

    def get_state(self):
        """the obervable state of the bank at each time step"""
        cf = self.bankmodel.calculate_cashflows("all")
        " Current interest rates"
        i = self.bankmodel.hullwhite.get_simulated_interest_rates(self.timestep - 1)
        z = self.bankmodel.hullwhite.get_simulated_zero_rates(self.timestep - 1)
        # liquidity
        liquidity = self.bankmodel.liquidity
        spread = z[-1] - z[0]
        state = {
            "liquidity": np.array([liquidity]),
            "cashflows": cf["cashflow"],
            "swap_rates": z.flatten(),
            "features": spread,
            "bank_rates": i.flatten(),
        }
        state = self.observation_space.normalize_observation(state)

        if not (self.observation_space.contains(state)):
            print(state)
            raise ValueError("State not in observation space")

        return state

    def reset(self, seed=None, options=None):
        # Reset the bank model
        if seed is not None:
            self.seed = seed
        self.bankmodel.reset()

        # Reset the episode length
        self.episode_length = EPISODE_LENGTH
        self.timestep = 0

        # Reset the statistics
        self.total_reward = 0
        self.total_nii = 0
        self.total_risk_penalty = 0
        self.total_liquidity_penalty = 0

        # data model to track a single episode
        self.df_model = None

        state = self.get_state()
        info = {}

        return state, info

    def reset_episode_statistics(self):
        self.episode_rewards = []
        self.episode_nii = []
        self.episode_risk_penalty = []
        self.episode_liquidity_penalty = []

    def close(self):
        super().close()
