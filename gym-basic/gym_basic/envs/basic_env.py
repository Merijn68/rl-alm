import gymnasium as gym

class BasicEnv(gym.Env):
    def __init__(self):
            self.action_space = gym.spaces.Discrete(5)
            self.observation_space = gym.spaces.Discrete(2)
    def step(self, action):
            state = 1
        
            if action == 2:
                reward = 1
            else:
                reward = -1
                
            truncated = False
            terminated = True
            info = {}
            return state, reward, terminated, truncated, info
    def reset(self):
            state = 0
            return state, []