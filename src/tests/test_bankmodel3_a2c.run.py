import gymnasium as gym
from stable_baselines3 import PPO
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_DIR))
from src.data.definitions import MODEL_PATH


def main():
    env_id = "gym_basic:bank-v3"
    model_name = "A2C_V_50000.zip"
    # Walk through the predictions from the trained model
    env = gym.make(env_id, render_mode="human")
    env = gym.wrappers.PassiveEnvChecker(env)
    model = PPO.load(Path(MODEL_PATH, model_name), env=env)
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
