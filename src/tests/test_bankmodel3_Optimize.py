import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import sys
from pathlib import Path
import time


def main():
    ROOT_DIR = Path(__file__).parents[2].absolute()
    sys.path.append(str(ROOT_DIR))

    from src.data.definitions import MODEL_PATH, TENSORBOARD_LOGS

    env_id = "gym_basic:bank-v3"
    env = gym.make(env_id, render_mode="human")
    env = gym.wrappers.PassiveEnvChecker(env)

    # model_name = "PPO_V_20230722-160348"
    model_name = "PPO_V_Optimize"

    env.set_render_output(model_name)
    model = PPO.load(Path(MODEL_PATH, model_name), env=env)
    env.reset()

    score = 0
    terminated = False
    truncated = False

    t = time.time()
    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward

    t1 = time.time()
    print(f"Time: {t1-t:.2f}")
    print(f"score: {score:.2f}")

    env.close()

    # evaluate the model
    env = gym.make(env_id, render_mode="human")
    env = gym.wrappers.PassiveEnvChecker(env)

    env.reset()
    result, n_steps = evaluate_policy(model, env, return_episode_rewards=True)
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=False,
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    env = gym.make(env_id, render_mode="human")
    env = gym.wrappers.PassiveEnvChecker(env)
    env.set_render_output(model_name)
    obs, info = env.reset()
    score = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _state = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
        # env.render()

    env.plot()
    env.close()

    print(f"reward={score:.2f}")


if __name__ == "__main__":
    main()
