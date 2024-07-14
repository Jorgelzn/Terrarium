import numpy as np
import pygame
from terrarium.env.src import constants as const

class Food:
    def __init__(self,  x, y, type):
        self.x = x
        self.y = y
        self.type = type
        if self.type == "water":
            self.sprite = pygame.image.load("../terrarium/env/data/food/pinap.png")
            self.sprite = pygame.transform.scale(self.sprite, (const.BLOCK_SIZE, const.BLOCK_SIZE))
        elif self.type == "land":
            self.sprite = pygame.image.load("../terrarium/env/data/food/razz.png")
            self.sprite = pygame.transform.scale(self.sprite, (const.BLOCK_SIZE, const.BLOCK_SIZE))
        else:
            raise Exception