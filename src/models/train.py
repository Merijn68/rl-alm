from typing import Tuple, List
import numpy as np
from stable_baselines3.common import type_aliases
from src.models.evaluate import evaluate_returns


def get_innermost_env(env):
    """Get innermost non-vectorized environment."""

    while hasattr(env, "env"):
        env = env.env
    return env


def train(
    model: "type_aliases.PolicyPredictor",
    env: "type_aliases.GymEnv",
    total_timesteps: int = 600,
    conf_level: float = 0.95,
    tb_log_name: str = "A2C",
) -> Tuple[float, float, float, float, List[float]]:
    """Trains a model for `total_timesteps` in the environment
    Measures the variability of the training results accross episodes
    """

    env.reset_episode_statistics()
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=tb_log_name,
    )
    episode_rewards = env.episode_rewards

    mean_reward, iqr_range, expected_shortfall, episode_rewards = evaluate_returns(
        episode_rewards, conf_level
    )

    return model, mean_reward, iqr_range, expected_shortfall, episode_rewards
