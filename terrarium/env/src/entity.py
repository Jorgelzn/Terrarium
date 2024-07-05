import numpy as np
from gymnasium.spaces import MultiDiscrete

class Entity:
    def __init__(self, agent_id, x, y, sprite, perception_range=2):
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
        for idx_line,obs_linne in enumerate(obs):
            for idx_col,obs_col in enumerate(obs_linne):
                y_dist = idx_line - center_y
                x_dist = idx_col - center_x

                pos_y = self.y + y_dist -1
                pos_x = self.x + x_dist -1

                if pos_x<0 or pos_y<0 or pos_y >= len(grid) or pos_x >= len(grid[0]):
                    obs[idx_line][idx_col] = -1
                else:
                    obs[idx_line][idx_col] = grid[pos_y][pos_x]

        return obs