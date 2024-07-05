import numpy as np
from gymnasium.spaces import MultiDiscrete

class Entity:
    def __init__(self, agent_id, x, y, sprite, perception_range=1):
        self.x = x
        self.y = y
        self.id = agent_id
        self.sprite = sprite
        self.perception_range = perception_range
        self.observation_space = MultiDiscrete(np.ones(shape=(2*self.perception_range+1,2*self.perception_range+1,2),dtype=np.int8))
        self.obs_ids = np.zeros((2*self.perception_range+1,2*self.perception_range+1,2),dtype=np.int8)
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
        grid[self.y][self.x] = 2


    def move_up(self):
        self.y -= 1

    def move_down(self):
        self.y += 1

    def move_left(self):
        self.x -= 1

    def move_right(self):
        self.x += 1

    def get_observation(self,terrain):
        terrain_len = len(terrain.terrain_type)
        obs = np.zeros((2*self.perception_range+1,2*self.perception_range+1,2))
        center_y = len(obs) % 2
        center_x = len(obs[0]) % 2
        for idx_line,obs_line in enumerate(obs):
            for idx_col,obs_col in enumerate(obs_line):
                y_dist = idx_line - center_y
                x_dist = idx_col - center_x

                pos_y = self.y + y_dist - self.perception_range + 1
                pos_x = self.x + x_dist - self.perception_range + 1

                if pos_x<0 or pos_y<0 or pos_y >= terrain_len or pos_x >= terrain_len:
                    obs[idx_line][idx_col][0] = -1
                    obs[idx_line][idx_col][1] = 0
                    self.obs_ids[idx_line][idx_col] = [-1,-1]
                else:
                    obs[idx_line][idx_col][0] = terrain.terrain_type[pos_y][pos_x]
                    obs[idx_line][idx_col][1] = terrain.agents[pos_y][pos_x]
                    self.obs_ids[idx_line][idx_col] = [pos_y,pos_x]


        return obs