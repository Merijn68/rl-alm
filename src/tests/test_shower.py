import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from src.data.definitions import MODEL_PATH
from pathlib import Path
from datetime import datetime

env = gym.make("gym_basic:shower-v1", render_mode="human")
env.set_render_output("random_walk")
obs, info = env.reset()

episodes = 10
# Perform random actions from the action space
for episode in range(1, episodes + 1):
    obs, info = env.reset()
    terminated = False
    truncated = False
    score = 0
    while not terminated and not truncated:
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward

print("Random_walk - Episode:{} Score:{}".format(episode, score))

# Random Agent, before training
model = DQN("MlpPolicy", "gym_basic:shower-v1", verbose=1)
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True,
)
print(f"DQN Untrained - mean_reward={mean_reward:.2f} +/- {std_reward}")

tensorboard_logs = "./tensorboard_logs/"


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


models = {
    "DGN": DQN(
        "MlpPolicy",
        "gym_basic:shower-v1",
        verbose=0,
        tensorboard_log=tensorboard_logs,
        learning_rate=linear_schedule(0.001),
    ),
    "PPO": PPO(
        "MlpPolicy",
        "gym_basic:shower-v1",
        verbose=0,
        tensorboard_log=tensorboard_logs,
        learning_rate=linear_schedule(0.001),
    ),
}

# Train the agent
for name, model in models.items():
    model.learn(total_timesteps=3e5, progress_bar=True)
    model.save(
        Path(MODEL_PATH, name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip")
    )
    env.set_render_output(name)
    obs, info = env.reset()
    score = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _state = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
        env.render()

    print(f"{name} Final score: {score}")

    # evaluate the model
    result, n_steps = evaluate_policy(model, env, return_episode_rewards=True)
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True,
    )
    print(f"{name} mean_reward={mean_reward:.2f} +/- {std_reward}")

env.close()
print("all done... That's all folks! ")
