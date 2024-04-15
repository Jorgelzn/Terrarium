# Example file showing a basic pygame "game loop"
import pygame
import numpy as np
import math
from gymnasium.spaces import Box,Discrete

class Obstacle():

    def __init__(self, initial_pos, size, color):
        self.pos = initial_pos
        self.color = color
        self.size = size
        self.collision_rect = pygame.Rect(self.pos[0]-self.size[0]/2,self.pos[1]-self.size[1]/2,self.size[0],self.size[1])

    def draw(self,screen):
        pygame.draw.rect(screen,self.color,self.collision_rect,1)
    

class Food():

    def __init__(self, initial_pos, radius, color):
        self.pos = initial_pos
        self.radius = radius
        self.color = color
        self.collision_rect = pygame.Rect(self.pos[0]-self.radius,self.pos[1]-self.radius,self.radius*2,self.radius*2)

    def draw(self,screen):
        #pygame.draw.rect(screen,"black",self.collision_rect,1)
        pygame.draw.circle(screen, self.color, self.pos, self.radius,0)

class Agent():

    def __init__(self, initial_pos, radius,velocity,direction,vision_len,color,agent_id):
        self.agent_id = agent_id
        self.pos = initial_pos
        self.radius = radius
        self.color = color
        self.collision_rect = pygame.Rect(self.pos[0]-self.radius,self.pos[1]-self.radius,self.radius*2,self.radius*2)
        self.velocity = velocity
        self.direction = np.deg2rad(direction)
        self.vision_len = vision_len
        self.vision = np.zeros((11,2))
        self.vision_color = np.empty((11), dtype=object)
        self.collision_distance = np.zeros((11),dtype=np.float32)
        self.obs = Box(low=-10, high=999, shape=(11,), dtype=np.float32)
        self.actions = ["up","down","left","right","turn_left","turn_right"]
        self.acts = Discrete(6)
        for idx,vision in enumerate(self.vision):
            self.vision[idx][0] = self.pos[0] + self.vision_len*math.cos(self.direction+np.deg2rad((idx-5)*10))
            self.vision[idx][1] = self.pos[1] - self.vision_len*math.sin(self.direction+np.deg2rad((idx-5)*10))
            self.vision_color[idx] = "black"
            self.collision_distance[idx] = 999
        self.dv = np.zeros(2)

    def draw(self,screen):
        #pygame.draw.rect(screen,"black",self.collision_rect,1)
        for idx,vision in enumerate(self.vision):
            pygame.draw.aaline(screen, self.vision_color[idx], self.pos,vision)
        pygame.draw.circle(screen, self.color, self.pos, self.radius,0)

    def update(self):
        self.pos += self.dv
        for idx,vision in enumerate(self.vision):
            self.vision[idx][0] = self.pos[0] + self.vision_len*math.cos(self.direction+np.deg2rad((idx-5)*10))
            self.vision[idx][1] = self.pos[1] - self.vision_len*math.sin(self.direction+np.deg2rad((idx-5)*10))
        self.collision_rect = pygame.Rect(self.pos[0]-self.radius,self.pos[1]-self.radius,self.radius*2,self.radius*2)

    def move(self,action,friction):
        movements=["up","down","right","left"]
        if action in movements:
            perpendicular=self.direction
            velocity_front = 0
            velocity_side = 0 
            if action=="up":
                velocity_front=self.velocity
            if action=="down":
                velocity_front=-self.velocity
            if action=="left":
                velocity_side=self.velocity
                perpendicular = self.direction+np.deg2rad(90)
            if action=="right":
                velocity_side=self.velocity
                perpendicular = self.direction-np.deg2rad(90)

            self.dv[0]=velocity_front*math.cos(self.direction) + velocity_side*math.cos(perpendicular)
            self.dv[1]=-velocity_front*math.sin(self.direction) - velocity_side*math.sin(perpendicular)

        if self.dv[0]>0:
            self.dv[0]-=friction
        elif self.dv[0]<0:
            self.dv[0]+=friction
        if self.dv[1]>0:
            self.dv[1]-=friction
        elif self.dv[1]<0:
            self.dv[1]+=friction

        if action=="turn_left":
            self.direction+=0.1
        if action=="turn_right":
            self.direction-=0.1
