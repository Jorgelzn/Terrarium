import noise
import pygame
import random
class Terrain:

    def __init__(self,world_size,block_size):
        self.draw_grid = []
        self.terrain_type = []
        self.objects = []
        self.terrain_textures = []
        self.water_anim = 0
        self.water_anim_timer = 0

        self.grass = []
        for i in range(4):
            grass = pygame.image.load("../terrarium/env/data/grass/grass_{}.png".format(i))
            self.grass.append(pygame.transform.scale(grass, (block_size, block_size)))

        self.water = []
        for i in range(8):
            water = pygame.image.load("../terrarium/env/data/water/water_{}.png".format(i))
            self.water.append(pygame.transform.scale(water, (block_size, block_size)))

        self.border_down = pygame.image.load("../terrarium/env/data/border.png")
        self.border_down = pygame.transform.scale(self.border_down,(block_size, block_size))
        self.border_up = pygame.transform.rotate(self.border_down,180)
        self.border_right = pygame.transform.rotate(self.border_down, 90)
        self.border_left = pygame.transform.rotate(self.border_down, 270)

        self.flower = pygame.image.load("../terrarium/env/data/flower.png")
        self.flower = pygame.transform.scale(self.flower, (block_size, block_size))

        scale = 0.001  # Adjust this for different terrain scales
        octaves = 8
        persistence = 2

        for y in range(0, world_size, block_size):
            self.draw_grid.append([])
            self.terrain_type.append([])
            self.objects.append([])
            self.terrain_textures.append([])
            for x in range(0, world_size, block_size):
                self.terrain_textures[-1].append([])
                self.draw_grid[-1].append(pygame.Rect(x, y, block_size, block_size))
                self.objects[-1].append(0)
                terrain_value = noise.pnoise2(x/block_size * scale, y/block_size * scale, octaves=octaves, persistence=persistence)
                if terrain_value > 0:
                    self.terrain_type[-1].append(0)
                    self.terrain_textures[-1][-1].append(self.grass[random.randrange(len(self.grass))])
                else:
                    self.terrain_type[-1].append(1)
                    self.terrain_textures[-1][-1].append(self.water[self.water_anim])

        #ADD DECORAION SPRITES TO LAND
        for idx_y,y in enumerate(self.terrain_type):
            for idx_x,x in enumerate(y):
                if x == 0:
                    if idx_y < len(self.terrain_type)-1 and self.terrain_type[idx_y+1][idx_x] == 1:
                        self.terrain_textures[idx_y][idx_x].append(self.border_down)
                    if idx_y > 0 and self.terrain_type[idx_y-1][idx_x] == 1:
                        self.terrain_textures[idx_y][idx_x].append(self.border_up)
                    if idx_x < len(y)-1 and self.terrain_type[idx_y][idx_x+1] == 1:
                        self.terrain_textures[idx_y][idx_x].append(self.border_right)
                    if idx_x > 0 and self.terrain_type[idx_y][idx_x-1] == 1:
                        self.terrain_textures[idx_y][idx_x].append(self.border_left)

                    if random.random() > 0.98:
                        self.terrain_textures[idx_y][idx_x].append(self.flower)