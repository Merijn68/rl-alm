import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases


def evaluate_returns(episode_rewards: List[float], conf_level: float = 0.95):
    """Calculate the mean, IQR, and CVaR of the episode rewards."""

    mean_reward = np.mean(episode_rewards)
    episode_rewards_sorted = episode_rewards.copy()
    episode_rewards_sorted.sort()

    # calculate dispersion of the policy performance across episodes
    q1 = np.percentile(episode_rewards_sorted, 25)
    q3 = np.percentile(episode_rewards_sorted, 75)
    iqr_range = q3 - q1

    # calculate the expected shortfall at the given confidence level
    # this is the average of the worst x% of episodes
    # e.g. if conf_level = 0.95, then we are looking at the worst 5% of episodes
    # and taking the average of those episodes
    tail = np.percentile(episode_rewards_sorted, conf_level)
    tail_values = [reward for reward in episode_rewards_sorted if reward < tail]
    expected_shortfall = np.mean(tail_values)
    return mean_reward, iqr_range, expected_shortfall, episode_rewards


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: gym.Env,
    n_eval_episodes: int = 10,
    conf_level: float = 0.95,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """

    Runs a fixed policy for ``n_eval_episodes`` episodes
    Measures the variability of the policy performance across episodes.

    This will return the mean reward - to measure the performance of the policy.
    This will return the IQR - to measure the dispersion of the policy performance across episodes.
    This will measure and return the Expected Shortfall at the given confidence level, which is the average of the worst x% of episodes.


    """

    episode_rewards = []
    current_episode = 0
    while current_episode < n_eval_episodes:
        score = 0
        terminated = False
        truncated = False
        obs, _ = env.reset()
        while not terminated and not truncated:
            action, _state = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            score = score + reward

        episode_rewards.append(score)
        current_episode += 1

    mean_reward, iqr_range, expected_shortfall, episode_rewards = evaluate_returns(
        episode_rewards, conf_level
    )

    return mean_reward, iqr_range, expected_shortfall, episode_rewards
