import numpy as np
from gymnasium.spaces import MultiDiscrete,Box
import pygame
from terrarium.env.src import constants as const

class Agent:
    def __init__(self, agent_id, x, y, type, perception_range=1):
        self.x = x
        self.y = y
        self.id = agent_id
        self.type = type
        if self.type == "water":
            self.sprite = pygame.image.load("../terrarium/env/data/mudkip.png")
            self.sprite = pygame.transform.scale(self.sprite,(const.BLOCK_SIZE, const.BLOCK_SIZE))
        elif self.type == "land":
            self.sprite = pygame.image.load("../terrarium/env/data/treecko.png")
            self.sprite = pygame.transform.scale(self.sprite, (const.BLOCK_SIZE, const.BLOCK_SIZE))
        else:
            raise Exception

        self.perception_range = perception_range
        #self.observation_space = Box(low=-1, high=1, shape=(2*self.perception_range+1,2*self.perception_range+1,2), dtype=int)
        self.observation_space = Box(low=-1, high=1, shape=((2 * self.perception_range + 1)**2*2,), dtype=int)
        self.obs_ids = np.zeros((2*self.perception_range+1,2*self.perception_range+1,2),dtype=int)


    def check_action(self,agents,terrain):
        if 0 < self.y < len(agents) and 0<self.x<len(agents) and agents[self.y][self.x]==0:     #GRID AND AGENTS CHECK

            if ((self.type == "land" and terrain[self.y][self.x] == 0) or           #LAND TYPE CHECK
                    (self.type == "water" and terrain[self.y][self.x] == 1)):
                return True
        else:
            return False

    def do_action(self,action,agents,terrain):

        previous_pos = (self.y, self.x)

        if action == 0:
            self.move_up()
        elif action == 1:
            self.move_down()
        elif action == 2:
            self.move_left()
        elif action == 3:
            self.move_right()

        if self.check_action(agents,terrain):
            agents[previous_pos[0]][previous_pos[1]] = 0
            agents[self.y][self.x] = 1
        else:
            self.y = previous_pos[0]
            self.x = previous_pos[1]


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
        obs = []#np.zeros(shape=(2*self.perception_range+1)**2*4)
        center_y = len(self.obs_ids) % 2
        center_x = len(self.obs_ids[0]) % 2
        for idx_line,obs_line in enumerate(self.obs_ids):
            for idx_col,obs_col in enumerate(self.obs_ids):
                y_dist = idx_line - center_y
                x_dist = idx_col - center_x

                pos_y = self.y + y_dist - self.perception_range + 1
                pos_x = self.x + x_dist - self.perception_range + 1

                if pos_x<0 or pos_y<0 or pos_y >= terrain_len or pos_x >= terrain_len:
                    obs.append(-1)
                    obs.append(0)
                    self.obs_ids[idx_line][idx_col] = [-1,-1]
                else:
                    obs.append(terrain.terrain_type[pos_y][pos_x])
                    obs.append(terrain.objects[pos_y][pos_x])
                    self.obs_ids[idx_line][idx_col] = [pos_y,pos_x]


        return obs