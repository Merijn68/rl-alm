""" Testing the A2C Model for the Bank Environment."""

import sys
import os
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import A2C
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


ROOT_DIR = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_DIR))
from src.models.bank_env import BankEnv
from src.data.definitions import MODEL_PATH, TENSORBOARD_LOGS


def get_innermost_env(env):
    """Get innermost non-vectorized environment."""

    while hasattr(env, "env"):
        env = env.env
    return env


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0."""
        return progress_remaining * initial_value

    return func


# class CustomLRScheduler:
#     def __init__(self, optimizer, scheduler):
#         self.optimizer = optimizer
#         self.scheduler = scheduler

#     def step(self):
#         self.scheduler.step()


def train_model(env, steps=60_000, log_dir="tmp/", ent_coef=0.01, model_name="A2C"):
    """Train the model"""

    model = A2C(
        "MlpPolicy",
        env,
        device="cpu",
        tensorboard_log=TENSORBOARD_LOGS,
        verbose=0,
        n_steps=60,
        learning_rate=linear_schedule(0.001),
        ent_coef=ent_coef,
    )

    # # Create an optimizer for the model's parameters
    # optimizer = optim.Adam(
    #     model.policy.parameters(), lr=0.001
    # )  # You can choose an optimizer and initial LR

    # # Create a PyTorch learning rate scheduler (e.g., StepLR)
    # scheduler = StepLR(
    #     optimizer, step_size=10000, gamma=0.5
    # )  # Adjust parameters as needed

    # # Create the custom learning rate scheduler
    # custom_scheduler = CustomLRScheduler(optimizer, scheduler)

    # for _ in range(steps):
    #     model.learn(
    #         total_timesteps=1,
    #         tb_log_name="a2c",
    #         progress_bar=True,
    #     )
    #     custom_scheduler.step()

    model.learn(
        total_timesteps=steps,
        tb_log_name=model_name,
        # progress_bar=True,
    )
    # custom_scheduler.step()

    modelpath = Path(
        MODEL_PATH,
        model_name + "_" + str(steps) + ".zip",
    )
    model.save(modelpath)
    return env, model  # remove to demonstrate saving and loading


def show_model(env_id, modelpath=MODEL_PATH / "A2C_V_50000.zip", model=None) -> int:
    """Show the model in the environment"""
    env = gym.make(env_id, render_mode="human")
    if model is None:
        model = A2C.load(path=modelpath, env=env)
    # env.set_render_output(modelpath.stem)
    obs, info = env.reset()
    score = 0
    terminated = False
    truncated = False

    env.set_render_output("A2C")

    while not terminated and not truncated:
        action, _state = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
        env.render()
    env.plot()
    env.close()
    return score


def random_walk(env_id):
    """Random walk in the environment"""
    env = gym.make(env_id, render_mode="human")

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
    env_id = "bank-v3"
    gym.register(id=env_id, entry_point=BankEnv, max_episode_steps=60)

    # Random walk in the environment
    random_walk(env_id)
    # Train the model
    train_model(env_id, steps=30_000)
    score = show_model(env_id)
    print("score: ", score)
    print("all done... That's all folks! ")


if __name__ == "__main__":
    main()
