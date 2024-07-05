import noise
import pygame

class Terrain:

    def __init__(self,world_size,block_size):
        self.draw_grid = []
        self.terrain_type = []
        self.agents = []
        scale = 0.0001  # Adjust this for different terrain scales
        octaves = 6
        persistence = 2

        for y in range(0, world_size, block_size):
            self.draw_grid.append([])
            self.terrain_type.append([])
            self.agents.append([])
            for x in range(0, world_size, block_size):
                self.draw_grid[-1].append(pygame.Rect(x, y, block_size, block_size))
                self.agents[-1].append(0)
                terrain_value = noise.pnoise2(x * scale, y * scale, octaves=octaves, persistence=persistence)
                if terrain_value > 0:
                    self.terrain_type[-1].append(0)
                else:
                    self.terrain_type[-1].append(1)