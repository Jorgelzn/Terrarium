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
velocity_constant=2
friction = 0.1
player_pos = np.array([screen.get_width() / 2, screen.get_height() / 2])
player_radius = 40
radar_len = 50
line_end =player_pos+radar_len


if __name__=="__main__": 

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        #MOVEMENT KEYS
        #keys = pygame.key.get_pressed()
        #if keys[pygame.K_w]:
        #    dv[1]=-velocity_constant
        #if keys[pygame.K_s]:
        #    dv[1]=velocity_constant
        #if keys[pygame.K_a]:
        #    dv[0]=-velocity_constant
        #if keys[pygame.K_d]:
        #    dv[0]=velocity_constant

        #DIRECTIONAL MOVEMENT

        #ANGLES

        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos[0]>player_pos[0]:
            dv[0]=velocity_constant
        if mouse_pos[0]<player_pos[0]:
            dv[0]=-velocity_constant
        if mouse_pos[1]>player_pos[1]:
            dv[1]=velocity_constant
        if mouse_pos[1]<player_pos[1]:
            dv[1]=-velocity_constant

        angle = np.arctan2(player_pos[1]-mouse_pos[1],mouse_pos[0]-player_pos[0])
        angle = np.rad2deg(angle)
        if angle<0:
            angle+=360
        print(angle)


        line_end[0] = player_pos[0] + math.cos(angle) * (mouse_pos[0] - player_pos[0]) - math.sin(angle) * (mouse_pos[1] - player_pos[1])
        line_end[1] = player_pos[1] + math.sin(angle) * (mouse_pos[0] - player_pos[0]) - math.cosgit st(angle) * (mouse_pos[1] - player_pos[1])
        #line_end[0] += np.cos(math.radians(angle)) * radar_len
        #line_end[1] += np.sin(math.radians(angle)) * radar_len

        #player_pos+=dv

        if dv[0]>0:
            dv[0]-=friction
        elif dv[0]<0:
            dv[0]+=friction
        if dv[1]>0:
            dv[1]-=friction
        elif dv[1]<0:
            dv[1]+=friction

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("white")

        # RENDER YOUR GAME HERE
        pygame.draw.circle(screen, "black", player_pos, player_radius,3)
        pygame.draw.aaline(screen, "black", player_pos,line_end)

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60



    pygame.quit()