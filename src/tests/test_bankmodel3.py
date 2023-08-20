# Lets try with vectorized environments
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback


from datetime import datetime
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_DIR))
from src.data.definitions import MODEL_PATH, TENSORBOARD_LOGS


class RewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, env: int, verbose=1):
        super(RewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.env = get_innermost_env(env)
        self.episode_rewards = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"Calls: {self.n_calls}")
            # Retrieve training reward
            # x, y = ts2xy(load_results(self.log_dir), "timesteps")
            episode = self.n_calls / self.env.episode_length
            reward = self.env.rewards[-1]
            if self.verbose > 0:
                print(f"Episode: {episode}")
                print(f"Reward: {reward:.2f}")
            self.episode_rewards.append(reward)
        return True


def make_env(env_id: str, rank: int, seed: int = 0):
    """Utility function for multiprocessed env."""

    def _init():
        env = gym.make(env_id, render_mode="human")
        # env = FrameStack(FlattenObservation(env), 4)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0."""
        return progress_remaining * initial_value

    return func


def get_innermost_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def main():
    env_id = "gym_basic:bank-v3"
    num_cpu = 1  # Number of processes to use - this does not work correct on windows

    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make(env_id, render_mode="human")
    env.set_render_output("Random")
    env.reset()

    score = 0
    terminated = False
    truncated = False
    steps = 0
    while not terminated and not truncated:
        steps += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
    env.close()
    print("steps: ", steps)
    print("score: ", score)

    # Create the environment
    env = gym.make(env_id, render_mode="human")

    # Create the callback: check every 60 steps
    # callback = RewardCallback(check_freq=60, log_dir=log_dir, env=env)

    tensorboard_logs = TENSORBOARD_LOGS

    # Set your desired values
    desired_total_steps = 300_000  # Set the total number of steps you want
    n_steps = 60  # Set the number of steps per iteration
    n_envs = 1  # Set the number of parallel environments

    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=tensorboard_logs,
        verbose=1,
        n_steps=n_steps,
        batch_size=n_steps * n_envs,
        learning_rate=linear_schedule(0.001),
        ent_coef=0.05,
    )

    # callback = RewardCallback()  # Set the callback's model attribute
    # callback.model = model

    model.learn(
        total_timesteps=desired_total_steps,
        # callback=callback,  # Register the callback here
        tb_log_name="ppo",
        progress_bar=True,
    )

    env = get_innermost_env(env)
    env.plot_rewards()

    modelpath = Path(
        MODEL_PATH,
        "PPO_V" + "_" + "NoLimit" + str(desired_total_steps) + ".zip",
    )
    model.save(modelpath)
    del model  # remove to demonstrate saving and loading

    env = gym.make(env_id, render_mode="human")
    model = PPO.load(path=modelpath, env=env)
    env.set_render_output(modelpath.stem)

    obs, info = env.reset()
    score = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _state = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
        # env.render()
    env.close()

    print("score: ", score)
    print("all done... That's all folks! ")


if __name__ == "__main__":
    main()
