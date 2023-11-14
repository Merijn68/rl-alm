# This model will simply use Duration Matching
import numpy as np

MAX_FUNDING_PER_TENOR = 1000
MIN_LIQ_AMOUNT = 1000


class Duration_matching:
    """Duration Matching Model"""

    def __init__(self, env):
        self.env = env
        self.name = "Duration Matching"
        self.bankmodel = env.bankmodel
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        """Predict the action"""
        assert self.observation_space.contains(obs)

        # Duration Matching
        # We can only determine where we want to invest.
        # If we borrow money for 5 years, we would need to repay the loan in 5 years.
        # This makes the most sence if income is also received in 5 years.
        # So we will invest where the income is highest
        obs = self.observation_space.denormalize_observation(obs)
        liq = obs["liquidity"]
        cf = obs["cashflows"]

        tenors = self.bankmodel.funding_tenors
        action = np.zeros(self.action_space.shape)
        for i, tenor in enumerate(tenors):
            if cf[tenor] <= 0:
                action[i] = 0
            else:
                action[i] = cf[tenor]

        if (sum(action) == 0) & (liq < MIN_LIQ_AMOUNT):
            for i, tenor in enumerate(tenors):
                action[i] = np.random.randint(0, MAX_FUNDING_PER_TENOR)

        action = np.clip(action, 0, MAX_FUNDING_PER_TENOR)
        action = np.array(action, dtype=np.float32)

        action = self.action_space.normalize_action(action)

        #      action = self.action_space.step(action)
        assert self.action_space.contains(action)

        return action, None
