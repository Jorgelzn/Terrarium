import numpy as np
from gymnasium.spaces import MultiDiscrete

class Entity:
    def __init__(self, agent_id, x, y, sprite, perception_range=1):
        self.x = x
        self.y = y
        self.id = agent_id
        self.sprite = sprite
        self.perception_range = perception_range
        self.observation_space = MultiDiscrete(np.full(shape=(2+self.perception_range,2+self.perception_range),fill_value=2))
    def do_action(self,action,grid):
        grid[self.y][self.x] = 0
        if action == 0:
            self.move_up()
        elif action == 1:
            self.move_down()
        elif action == 2:
            self.move_left()
        elif action == 3:
            self.move_right()
        grid[self.y][self.x] = 1


    def move_up(self):
        self.y -= 1

    def move_down(self):
        self.y += 1

    def move_left(self):
        self.x -= 1

    def move_right(self):
        self.x += 1

    def get_observation(self,grid):
        obs = np.zeros((2*self.perception_range+1,2*self.perception_range+1))
        center_y = len(obs) % 2
        center_x = len(obs[0]) % 2
        y_minus = self.y-self.perception_range if self.y-self.perception_range >= 0 else None
        y_plus = self.y+self.perception_range if self.y+self.perception_range < len(grid) else None
        x_minus = self.x - self.perception_range if self.x - self.perception_range >= 0 else None
        x_plus = self.x + self.perception_range if self.x + self.perception_range < len(grid[0]) else None
        for obs_y in range(len(obs)):
            for obs_x in range(len(obs[obs_y])):
                obs[obs_y][obs_x] = grid[obs_y][obs_x]

        return obs