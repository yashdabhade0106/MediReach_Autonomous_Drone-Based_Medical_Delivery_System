import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneNavEnv(gym.Env):
    """
    Custom 2D Grid Environment for Drone Navigation with Obstacles and Wind.
    State space: [drone_x, drone_y, target_x, target_y]
    Action space: Discrete 4 (up, down, left, right)
    """
    def __init__(self, grid_size=10):
        super(DroneNavEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(4,), dtype=np.int32)
        
        self.drone_pos = [0, 0]
        self.target_pos = [grid_size-1, grid_size-1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = [0, 0]
        self.target_pos = [self.grid_size-1, self.grid_size-1]
        return np.array(self.drone_pos + self.target_pos, dtype=np.int32), {}

    def step(self, action):
        if action == 0: # up
            self.drone_pos[1] = min(self.drone_pos[1] + 1, self.grid_size - 1)
        elif action == 1: # down
            self.drone_pos[1] = max(self.drone_pos[1] - 1, 0)
        elif action == 2: # left
            self.drone_pos[0] = max(self.drone_pos[0] - 1, 0)
        elif action == 3: # right
            self.drone_pos[0] = min(self.drone_pos[0] + 1, self.grid_size - 1)
            
        done = bool(self.drone_pos == self.target_pos)
        reward = 10.0 if done else -0.1
        
        return np.array(self.drone_pos + self.target_pos, dtype=np.int32), reward, done, False, {}

    def render(self):
        pass
