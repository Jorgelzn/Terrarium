
import pygame
from terrarium.env.src import constants as const

class Camera:
    def __init__(self, world_size):

        self.world_size = world_size
        self.width = const.SCREEN_WIDTH
        self.height = const.SCREEN_HEIGHT
        self.camera = pygame.Rect(-self.world_size/2+self.width/2, -self.world_size/2+self.height/2
                                , self.width, self.height)
        self.dragging = False
        self.mouse_start = None

    def apply(self, entity):
        return entity.move(self.camera.topleft)

    def start_drag(self, mouse_pos):
        self.dragging = True
        self.mouse_start = mouse_pos

    def stop_drag(self):
        self.dragging = False
        self.mouse_start = None

    def update_drag(self, mouse_pos):
        if self.dragging and self.mouse_start:
            dx = mouse_pos[0] - self.mouse_start[0]
            dy = mouse_pos[1] - self.mouse_start[1]
            self.camera.x += dx
            self.camera.y += dy
            self.mouse_start = mouse_pos

            # Limit scrolling to map size
            if self.camera.x > 0:
                self.camera.x = 0
            elif self.camera.x < -self.world_size+self.height:
                self.camera.x = -self.world_size+self.height
            if self.camera.y > 0:
                self.camera.y = 0
            elif self.camera.y < -self.world_size+self.height:
                self.camera.y = -self.world_size+self.height