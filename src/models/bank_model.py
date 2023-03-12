from gymnasium import Env
import numpy as np
import random

# Bank environment for ALM Agent to train
# Bank environments will be made progressively more complex starting with a very simplified model
class BankEnv_01(Env):
    def __init__(self):
        
        # Actions we can take: 0 (sell), 1 do noting, 2 buy
        self.action_space = Discrete(3)
        
        # Division of assets array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))

        # Set start state
        self.state = 0.5 + random.randint(-0.3,0.3)

        # Set episode length 
        self.episode_length = 60
	self.timestep = 0

        
    def step(self, action):
        # Apply action
        self.state += action - 1 
        # Reduce episode length by 1 second
	self.timestep += 1
        self.episode_length -= 1 
        
        # Calculate risk and reward

	short_term_interest = get_interest(self.timestep, 'ST')
	long_term_interest = get_interest(self.timestep, 'LT')

	short_term_interest * state + long_term_interest * (1- state) 

        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        
        # Check if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self, mode):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = 38 + random.randint(-3,3)
        # Reset shower time
        self.shower_length = 60 
        return self.state

def get_interest(timestep, class):
	if class == 'LT':
		interest = long_term_interest
	else:
		interest = short_term_interest
	end
	return interest[timestep]
	

