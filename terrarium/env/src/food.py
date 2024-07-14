import numpy as np

class Food:
    def __init__(self, agent_id, x, y, sprite):
        self.x = x
        self.y = y
        self.id = agent_id
        self.sprite = sprite
