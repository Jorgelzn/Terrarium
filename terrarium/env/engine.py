# Example file showing a basic pygame "game loop"
import pygame
import numpy as np
import math

pygame.init()
screen = pygame.display.set_mode((1280, 500))
clock = pygame.time.Clock()
running = True
friction = 0.1

class Agent():

    def __init__(self, initial_pos, radius,velocity,direction,vision_len):
        self.pos = initial_pos
        self.velocity = velocity
        self.direction = np.deg2rad(direction)
        self.radius = radius
        self.vision_len = vision_len
        self.vision = np.zeros(2)
        self.vision[0] = self.pos[0] + self.vision_len*math.cos(self.direction)
        self.vision[1] = self.pos[1] - self.vision_len*math.sin(self.direction)
        self.vision_color = "black"
        self.agent_color = "black"
        self.collision_rect = pygame.Rect(self.pos[0]-self.radius,self.pos[1]-self.radius,self.radius*2,self.radius*2)
        self.collision_distance = 999
        self.dv = np.zeros(2)

    def draw(self):
        pygame.draw.circle(screen, self.agent_color, self.pos, self.radius,3)
        pygame.draw.aaline(screen, self.vision_color, self.pos,self.vision)
        pygame.draw.rect(screen,"black",self.collision_rect,1)

    def update(self):
        self.pos += self.dv
        self.vision[0] = self.pos[0] + self.vision_len*math.cos(self.direction)
        self.vision[1] = self.pos[1] - self.vision_len*math.sin(self.direction)
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

    adan = Agent(np.array([screen.get_width() / 2, screen.get_height() / 2]),40,4,90,200)

    while running:
        # fill the screen with a color to wipe away anything from last frame
        screen.fill("white")
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        obstacle = pygame.draw.rect(screen, "black",pygame.Rect(200, 100, 200, 100),1)

        #MOVEMENT ACTIONS
        keys = pygame.key.get_pressed()
        block_collide = obstacle.colliderect(adan.collision_rect)
        if not block_collide:
            adan.move(keys)
        else:
            adan.dv = -adan.dv

        adan.update()

        line_color = "black"
        line_collide = obstacle.clipline(adan.pos,adan.vision)
        if line_collide:
            adan.vision_color="red"
            adan.vision=np.array(line_collide[0])
            adan.collision_distance = np.sqrt((adan.pos[0]-adan.vision[0])**2+(adan.pos[1]-adan.vision[1])**2)-adan.radius
        else:
            adan.vision_color="black"

        # RENDER YOUR GAME HERE
        adan.draw()

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60



    pygame.quit()