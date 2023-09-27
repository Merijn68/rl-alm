import gymnasium as gym
from stable_baselines3 import A2C
import sys
from pathlib import Path

root_dir = Path(__file__).parents[2].absolute()
sys.path.append(str(root_dir))
print(root_dir)

from src.data.definitions import MODEL_PATH
from src.models.bank_env import BankEnv

def main():
    env_id = "bank-v3"
    model_name = "A2C_V_50000.zip"
    
    gym.register(id=env_id, entry_point=BankEnv, max_episode_steps=60)    
    
    # Walk through the predictions from the trained model
    env = gym.make(env_id, render_mode="human")
    env = gym.wrappers.PassiveEnvChecker(env)
    model = A2C.load(Path(MODEL_PATH, model_name), env=env)
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
