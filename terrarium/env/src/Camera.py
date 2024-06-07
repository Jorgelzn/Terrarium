
import pygame

class Camera:
    def __init__(self, x, y, width, height,world_size):
        self.camera = pygame.Rect(x, y, width, height)
        self.world_size = world_size
        self.width = width
        self.height = height
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
            if self.camera.x < 0:
                self.camera.x = 0
            elif self.camera.x > self.world_size:
                self.camera.x = self.world_size
            if self.camera.y < 0:
                self.camera.y = 0
            elif self.camera.y > self.world_size:
                self.camera.y = self.world_size