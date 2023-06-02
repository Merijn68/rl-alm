import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from src.data.definitions import MODEL_PATH

from pathlib import Path

env = gym.make("gym_basic:shower-v1", render_mode="human")
model = PPO.load(Path(MODEL_PATH, "PPO_V_20230520-123537"), env=env)

# evaluate the model
result, n_steps = evaluate_policy(model, env, return_episode_rewards=True)
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=False,
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

env.set_render_output("PPO_V_20230520-123537")
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
