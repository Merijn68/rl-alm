import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# env = gym.make("gym_basic:basic-v2", render_mode="human")
env = gym.make("gym_basic:shower-v1", render_mode="human")
obs, info = env.reset()
print(obs)

check_env(env, warn=True)
