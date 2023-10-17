import gymnasium as gym
import numpy as np

MAX_FUNDING_PER_TENOR = 1000


# class ActionSpace(gym.spaces.Box):
#     def __init__(self, num_tenors, min_units: int = -1, max_units: int = 1, seed=None):
#         low = np.array([min_units] * num_tenors, dtype=np.float32)
#         high = np.array([max_units] * num_tenors, dtype=np.float32)
#         super().__init__(
#             low=np.float32(low),
#             high=np.float32(high),
#             shape=(num_tenors,),
#             dtype=np.float32,
#         )
#         self.scaling_factor = MAX_FUNDING_PER_TENOR

#     # Function to normalize actions to [0, 1] and then map to the investment range
#     def normalize_action(self, action):
#         return (((action) / (self.scaling_factor / 2)) - 1).astype(np.float32)

#     # Function to denormalize actions to the original range
#     def denormalize_action(self, normalized_action):
#         return (np.round((normalized_action + 1) * (self.scaling_factor / 2))).astype(
#             np.float32
#         )


class ActionSpace(gym.spaces.Box):
    def __init__(self, num_tenors, min_units: int = 0, max_units: int = 1, seed=None):
        low = np.array([min_units] * num_tenors, dtype=np.float32)
        high = np.array([max_units] * num_tenors, dtype=np.float32)
        super().__init__(
            low=np.float32(low),
            high=np.float32(high),
            shape=(num_tenors,),
            dtype=np.float32,
        )

    def normalize_action(self, action):
        return action / MAX_FUNDING_PER_TENOR

    # Function to denormalize actions to the original range
    def denormalize_action(self, normalized_action):
        return normalized_action * MAX_FUNDING_PER_TENOR


def main():
    action_space = ActionSpace(5)
    print("Action Space:", action_space)

    sample_action = action_space.sample()
    allocation = np.array([0.1, 0.1, 0.0, 0.9, 0.1], dtype=np.float32)
    sample_action = action_space.denormalize_action(allocation)
    print("Sampled Allocation:", allocation)
    print("Sampled Action:", sample_action)


if __name__ == "__main__":
    main()
