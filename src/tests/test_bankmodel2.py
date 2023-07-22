# Lets try with vectorized environments
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from datetime import datetime
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_DIR))

from src.data.definitions import MODEL_PATH, TENSORBOARD_LOGS


def make_env(env_id: str, rank: int, seed: int = 0):
    """Utility function for multiprocessed env."""

    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0."""
        return progress_remaining * initial_value

    return func


def main():
    env_id = "gym_basic:bank-v2"
    num_cpu = 2  # Number of processes to use

    env = gym.make(env_id, render_mode="human")
    env.set_render_output("Random")
    env.reset()
    sample = env.observation_space.sample()
    print(env.action_space.nvec)

    score = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
        # env.render()
    env.close()
    print("score: ", score)

    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, rank=i + 1) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    tensorboard_logs = TENSORBOARD_LOGS

    model = PPO(
        "MlpPolicy",
        vec_env,
        tensorboard_log=tensorboard_logs,
        verbose=1,
        learning_rate=linear_schedule(0.001),
    )

    model.learn(total_timesteps=3e5, progress_bar=True)  # 300_000 steps
    modelpath = Path(
        MODEL_PATH,
        "PPO_V" + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip",
    )
    model.save(modelpath)
    del model  # remove to demonstrate saving and loading

    env = gym.make(env_id, render_mode="human")
    model = PPO.load(path=modelpath, env=env)
    env.set_render_output(modelpath.stem)

    obs, info = env.reset()
    score = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _state = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
        # env.render()
    env.close()

    print("score: ", score)

    print("all done... That's all folks! ")


if __name__ == "__main__":
    main()
