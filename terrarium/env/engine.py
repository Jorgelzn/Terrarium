# Example file showing a basic pygame "game loop"
import pygame
import numpy as np
import math
# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 500))
clock = pygame.time.Clock()
running = True
dv = np.zeros(2)
velocity_constant=10
friction = 0.1
player_pos = np.array([screen.get_width() / 2, screen.get_height() / 2])
player_radius = 40
radar_len = 100
direction = 0
line_end =player_pos+radar_len


if __name__=="__main__": 

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        velocity_front=0
        velocity_side=0
        perpendicular=direction
        rad_front = np.deg2rad(direction)
        rad_side= np.deg2rad(perpendicular)

        line_end[0] = player_pos[0] + radar_len*math.cos(rad_front)
        line_end[1] = player_pos[1] - radar_len*math.sin(rad_front)

        #MOVEMENT ACTIONS
        keys = pygame.key.get_pressed()
        movements = [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d,]
        if True in [keys[k] for k in movements]:
            if keys[pygame.K_w]:
                velocity_front=velocity_constant
            if keys[pygame.K_s]:
                velocity_front=-velocity_constant
            if keys[pygame.K_a]:
                velocity_side=velocity_constant
                perpendicular = direction+90
            if keys[pygame.K_d]:
                velocity_side=velocity_constant
                perpendicular = direction-90

            dv[0]=velocity_front*math.cos(rad_front) + velocity_side*math.cos(rad_side)
            dv[1]=-velocity_front*math.sin(rad_front) - velocity_side*math.sin(rad_side)
        else:
            if keys[pygame.K_LEFT]:
                direction+=1
            if keys[pygame.K_RIGHT]:
                direction-=1
            if dv[0]>0:
                dv[0]-=friction
            elif dv[0]<0:
                dv[0]+=friction
            if dv[1]>0:
                dv[1]-=friction
            elif dv[1]<0:
                dv[1]+=friction

        player_pos+=dv

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("white")

        # RENDER YOUR GAME HERE
        pygame.draw.circle(screen, "black", player_pos, player_radius,3)
        pygame.draw.aaline(screen, "black", player_pos,line_end)

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60



    pygame.quit()