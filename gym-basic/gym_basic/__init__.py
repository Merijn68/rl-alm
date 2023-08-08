from gymnasium.envs.registration import register

# register(id='basic-v0',entry_point='gym_basic.envs:BasicEnv',)
register(
    id="gym-basic/basic-v0",
    entry_point="gym_basic.envs:basic",
    max_episode_steps=300,
)
register(
    id="basic-v2",
    entry_point="gym_basic.envs:BasicEnv2",
)
register(
    id="shower-v1",
    entry_point="gym_basic.envs:ShowerEnv",
    max_episode_steps=300,
)
register(
    id="bank-v1",
    entry_point="gym_basic.envs:BankEnv",
    max_episode_steps=252,
)
register(
    id="bank-v2",
    entry_point="gym_basic.envs:BankEnv2",
    max_episode_steps=252,
)
register(
    id="bank-v3",
    entry_point="gym_basic.envs:BankEnv3",
    max_episode_steps=252,
)
