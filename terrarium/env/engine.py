# Example file showing a basic pygame "game loop"
import pygame
import numpy as np
import math

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

    adan = Agent(np.array([screen.get_width() / 2, screen.get_height() / 2]),40,4,90,200,"black")
    elements = [Obstacle(np.array([600,400]), np.array([200,100]), "black"),Food(np.array([400,400]),30,"green")]

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