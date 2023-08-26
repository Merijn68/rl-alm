""" Testing the A2C Model for the Bank Environment."""

import sys
import os
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import A2C

ROOT_DIR = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_DIR))
from src.data.definitions import MODEL_PATH, TENSORBOARD_LOGS


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0."""
        return progress_remaining * initial_value

    return func


def get_innermost_env(env):
    """Get innermost non-vectorized environment."""

    while hasattr(env, "env"):
        env = env.env
    return env


def train_model(env_id, steps=50_000, log_dir="tmp/"):
    """Train the model"""

    # Create log dir
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make(env_id, render_mode="human")
    model = A2C(
        "MlpPolicy",
        env,
        device="cpu",
        tensorboard_log=TENSORBOARD_LOGS,
        verbose=1,
        n_steps=60,
        # batch_size=n_steps * n_envs,
        learning_rate=linear_schedule(0.001),
        ent_coef=0.05,
    )

    model.learn(
        total_timesteps=steps,
        tb_log_name="ppo",
        progress_bar=True,
    )

    env = get_innermost_env(env)
    env.plot_rewards()

    modelpath = Path(
        MODEL_PATH,
        "A2C_V" + "_" + str(steps) + ".zip",
    )
    model.save(modelpath)
    del model  # remove to demonstrate saving and loading


def show_model(env_id, modelpath=MODEL_PATH / "A2C_V_50000.zip") -> int:
    """Show the model in the environment"""
    env = gym.make(env_id, render_mode="human")
    model = A2C.load(path=modelpath, env=env)
    env.set_render_output(modelpath.stem)
    obs, info = env.reset()
    score = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _state = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
    env.close()
    return score


def random_walk(env_id):
    """Random walk in the environment"""
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


def main():
    env_id = "gym_basic:bank-v3"
    # Random walk in the environment
    random_walk(env_id)
    # Train the model
    train_model(env_id, steps=30_000)
    score = show_model(env_id)
    print("score: ", score)
    print("all done... That's all folks! ")


if __name__ == "__main__":
    main()
