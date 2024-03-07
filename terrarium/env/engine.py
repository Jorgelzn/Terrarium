# Example file showing a basic pygame "game loop"
import pygame
import numpy as np
import math
import random

pygame.init()
screen = pygame.display.set_mode((1280, 500))
clock = pygame.time.Clock()
running = True
friction = 0.1


class Entity():
    
    def draw():
        pass 

class Obstacle(Entity):

    def __init__(self, initial_pos, size, color):
        self.pos = initial_pos
        self.color = color
        self.size = size
        self.collision_rect = pygame.Rect(self.pos[0]-self.size[0]/2,self.pos[1]-self.size[1]/2,self.size[0],self.size[1])

    def draw(self):
        pygame.draw.rect(screen,self.color,self.collision_rect,1)
    

class Food(Entity):

    def __init__(self, initial_pos, radius, color):
        self.pos = initial_pos
        self.radius = radius
        self.color = color
        self.collision_rect = pygame.Rect(self.pos[0]-self.radius,self.pos[1]-self.radius,self.radius*2,self.radius*2)

    def draw(self):
        #pygame.draw.rect(screen,"black",self.collision_rect,1)
        pygame.draw.circle(screen, self.color, self.pos, self.radius,0)

class Agent(Entity):

    def __init__(self, initial_pos, radius,velocity,direction,vision_len,color):
        self.pos = initial_pos
        self.radius = radius
        self.color = color
        self.collision_rect = pygame.Rect(self.pos[0]-self.radius,self.pos[1]-self.radius,self.radius*2,self.radius*2)
        self.velocity = velocity
        self.direction = np.deg2rad(direction)
        self.vision_len = vision_len
        self.vision = np.zeros((11,2))
        self.vision_color = np.empty((11), dtype=object)
        self.collision_distance = np.zeros((11))
        for idx,vision in enumerate(self.vision):
            self.vision[idx][0] = self.pos[0] + self.vision_len*math.cos(self.direction+np.deg2rad((idx-5)*10))
            self.vision[idx][1] = self.pos[1] - self.vision_len*math.sin(self.direction+np.deg2rad((idx-5)*10))
            self.vision_color[idx] = "black"
            self.collision_distance[idx] = 999
        self.dv = np.zeros(2)

    def draw(self):
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

    def move(self,keys):
        movements=[pygame.K_w,pygame.K_s,pygame.K_a,pygame.K_d]
        if True in [keys[k] for k in movements]:
            perpendicular=self.direction
            velocity_front = 0
            velocity_side = 0 
            if keys[pygame.K_w]:
                velocity_front=self.velocity
            if keys[pygame.K_s]:
                velocity_front=-self.velocity
            if keys[pygame.K_a]:
                velocity_side=self.velocity
                perpendicular = self.direction+np.deg2rad(90)
            if keys[pygame.K_d]:
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

        if keys[pygame.K_LEFT]:
            self.direction+=0.1
        if keys[pygame.K_RIGHT]:
            self.direction-=0.1

if __name__=="__main__":

    environment = {
        "agents":1,
        "obstacles":10,
        "food":5
    }

    elements = []

    for obj_def in environment:
        for obj_num in range(environment[obj_def]):
            created = False
            while not created:
                pos = np.array([random.uniform(0,screen.get_width()), random.uniform(0,screen.get_height())])
                if obj_def ==  "agents":
                    obj = Agent(pos,40,4,random.randint(0, 360),200,"black")
                elif obj_def == "obstacles":
                    obj = Obstacle(pos, np.array([random.uniform(0,300),random.uniform(0,300)]), "black")
                elif obj_def == "food":
                    obj = Food(pos,30,"green")
                collisions = [elem for elem in elements if elem.collision_rect.colliderect(obj.collision_rect)]
                if not collisions:
                        elements.append(obj)
                        created = True

    adan = elements.pop(0)

    while running:
        # fill the screen with a color to wipe away anything from last frame
        screen.fill("white")
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        keys = pygame.key.get_pressed()

        collisions = [idx for idx,elem in enumerate(elements) if elem.collision_rect.colliderect(adan.collision_rect)]
        if not collisions:
            adan.move(keys)
        else:
            bounce = False
            for c in collisions:
                if type(elements[c]) is Food:
                    elements.pop(c)
                else:
                    bounce = True
            if bounce:
                adan.dv = -adan.dv

        adan.update()

        for idx,vision in enumerate(adan.vision):
            for elem in elements:
                line_collide = elem.collision_rect.clipline(adan.pos,vision)
                if line_collide:
                    adan.vision_color[idx]="red"
                    adan.vision[idx]=np.array(line_collide[0])
                    adan.collision_distance[idx] = np.sqrt((adan.pos[0]-vision[0])**2+(adan.pos[1]-vision[1])**2)-adan.radius
                    break
                else:
                    adan.vision_color[idx]="black"
                    adan.collision_distance[idx]=999

        # RENDER YOUR GAME HERE

        for elem in elements:
            elem.draw()
        adan.draw()
        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60



    pygame.quit()