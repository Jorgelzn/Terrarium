import pygame
import sys
from pygame.locals import *

pygame.init()

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

SCREEN = pygame.display.set_mode(WINDOW_SIZE)
SCREEN.fill((100, 100, 100))
zoom = 2
display = pygame.Surface((int(WINDOW_SIZE[0] / zoom), int(WINDOW_SIZE[1] / zoom)))

block_size = 20

def drawGrid():
    for x in range(0, WINDOW_WIDTH, block_size):
        for y in range(0, WINDOW_HEIGHT, block_size):
            rect = pygame.Rect(x, y, block_size, block_size)
            pygame.draw.rect(SCREEN, (200, 200, 200), rect, 3)

def quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

# Main game loop
while True:
    quit()
    drawGrid()
    pygame.display.update()
