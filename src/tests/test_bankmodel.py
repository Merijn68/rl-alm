import pytest
import pandas as pd
from dateutil.parser import parse
from pathlib import Path
from datetime import datetime
import gymnasium as gym
from src.models.action_space import ActionSpace
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from src.models.bank_model import Bankmodel
from src.data.definitions import DATA_DEBUG, DEBUG_MODE, MODEL_PATH
from timeit import default_timer as timer

TENSORBOARD_LOGS = "./tensorboard_logs/"


def make_env(env_id: str, rank: int, seed: int = 0, bm: Bankmodel = None):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        env = gym.make(env_id, bm=bm, render_mode="human")
        return env

    set_random_seed(seed)
    return _init


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def main():
    pd.options.display.float_format = "{:.2f}".format

    env_id = "gym_basic:bank-v1"

    bankmodel = Bankmodel()
    start_pos_date = bankmodel.pos_date
    bankmodel.generate_mortgage_contracts(n=100, amount=1000000)
    action_space = ActionSpace()
    env = gym.make(env_id, bm=bankmodel, render_mode="human")

    state, info = env.reset()
    score = 0
    terminated = False
    truncated = False
    t0 = timer()
    while not terminated and not truncated:
        action = action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
        env.render()
    env.close()
    t1 = timer()
    print(f"Reward = {score}")
    print(f"Time to run {t1-t0} seconds")
    print("all done... That's all folks! ")

    # Random Walk Buy or sell random swaps at each step
    t0 = timer()

    for i in range(252):
        # Take random actions in the environment
        action = action_space.sample()
        action = action_space.translate_action(action)
        bankmodel.apply_action(*action)
        bankmodel.step(1)
        bankmodel.fixing_interest_rate_swaps()
    t1 = timer()
    print(f"Time to run {t1-t0} seconds")

    # Reward need to be setup. For now it is only the nii
    print(f"start date = {start_pos_date}, end date = {bankmodel.pos_date}")
    nii = bankmodel.calculate_nii()
    print("nii = ", nii)
    npv = bankmodel.calculate_npv(bankmodel.df_cashflows, bankmodel.pos_date)
    print("npv = ", sum(npv["npv"]))
    bpv = bankmodel.calculate_bpv()
    print("bpv = ", bpv)

    # if DEBUG_MODE:
    #    bankmodel.df_cashflows.to_excel(Path(DATA_DEBUG, "df_cashflows.xlsx"))
    #    npv.to_excel(Path(DATA_DEBUG, "npv.xlsx"))
    #    bpv.to_excel(Path(DATA_DEBUG, "bpv.xlsx"))

    # Now lets see if a RL model can do better

    num_cpu = 1
    vec_env = SubprocVecEnv(
        [make_env(env_id, rank=i, bm=bankmodel) for i in range(num_cpu)]
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        tensorboard_log=TENSORBOARD_LOGS,
        verbose=1,
        learning_rate=linear_schedule(0.001),
    )
    model.learn(total_timesteps=100, progress_bar=True)  # 3e5
    modelpath = Path(
        MODEL_PATH,
        "PPO_V" + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip",
    )
    model.save(modelpath)

    # env = gym.make(env_id, bankmodel=bankmodel, render_mode="human")
    env.reset()
    env = gym.make(env_id, bm=bankmodel, render_mode="human")

    env.set_render_output(modelpath.stem)

    obs, info = env.reset()
    score = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _state = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
        env.render()
    env.close()

    print("all done... That's all folks! ")


if __name__ == "__main__":
    main()
