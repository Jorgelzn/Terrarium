import random
from copy import copy

import numpy as np
from gymnasium.spaces import Box,Discrete

from pettingzoo import ParallelEnv

from pettingzoo.test import parallel_api_test

import pygame
from engine import Food,Agent,Obstacle

class Terrarium(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "Terrarium",
    }

    def __init__(self,settings):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - escape x and y coordinates
        - guard x and y coordinates
        - prisoner x and y coordinates
        - timestamp
        - possible_agents

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """

        self.escape_y = 0
        self.escape_x = 0
        self.guard_y = 0
        self.guard_x = 0
        self.prisoner_y = 0
        self.prisoner_x = 0
        self.timestep = 0
        self.possible_agents = ["animal_"+str(idx) for idx in range(settings["agents"])]
        self.screen = pygame.display.set_mode((1280, 500))
        self.clock = pygame.time.Clock()
        self.running = True
        self.friction = 0.1
        self.settings = settings

        self.acts = Discrete(6)
        self.obs = Box(low=0, high=999, shape=(11,), dtype=np.float32)

        self.elements = []
        for obj_def in self.settings:
            for obj_num in range(settings[obj_def]):
                created = False
                while not created:
                    pos = np.array([random.uniform(0,self.screen.get_width()), random.uniform(0,self.screen.get_height())])
                    if obj_def ==  "agents":
                        obj = Agent(pos,40,4,random.randint(0, 360),200,"black")
                    elif obj_def == "obstacles":
                        obj = Obstacle(pos, np.array([random.uniform(0,300),random.uniform(0,300)]), "black")
                    elif obj_def == "food":
                        obj = Food(pos,30,"green")
                    collisions = [elem for elem in self.elements if elem.collision_rect.colliderect(obj.collision_rect)]
                    if not collisions:
                            self.elements.append(obj)
                            created = True
        self.init_elements = copy(self.elements)

        pygame.init()

    def step(self,actions):
        # clean screen
        self.screen.fill("white")
        agents = self.elements[0:self.settings["agents"]]
        for idx,agent in enumerate(agents):
            collisions = [idx for idx,elem in enumerate(self.elements) if elem.collision_rect.colliderect(agent.collision_rect) and elem!=agent]

            if not collisions:
                agent.move(agent.actions[actions["animal_"+str(idx)]],self.friction)
            else:
                bounce = False
                for c in collisions:
                    if type(self.elements[c]) is Food:
                        self.elements.pop(c)
                    else:
                        bounce = True
                if bounce:
                    agent.dv = -agent.dv

            agent.update()

            for idx,vision in enumerate(agent.vision):
                collided = False
                for elem in self.elements:
                    if elem!=agent:
                        line_collide = elem.collision_rect.clipline(agent.pos,vision) 
                        if line_collide:
                            agent.vision_color[idx]="red"
                            agent.vision[idx]=np.array(line_collide[0])
                            agent.collision_distance[idx] = np.sqrt((agent.pos[0]-vision[0])**2+(agent.pos[1]-vision[1])**2)-agent.radius
                            collided=True
                if not collided:
                    agent.vision_color[idx]="black"
                    agent.collision_distance[idx]=999
                    
        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if Food not in [type(elem) for elem in self.elements]:
            rewards = {a:1 for a in self.agents}
            terminations = {a: True for a in self.agents}
            self.agents = []
            self.running = False


        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100000:
            rewards = {a:0 for a in self.agents}
            truncations = {a: True for a in self.agents}
            self.agents = []
        self.timestep += 1

        # Get observations
        observations = {"animal_"+str(idx):{"observation":agent.collision_distance} for idx,agent in enumerate(agents)}


        # Get dummy infos (not used in this example)
        infos = {a:{} for a in self.agents}

        return observations, rewards, terminations, truncations, infos


    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.elements = copy(self.init_elements)
        agents = self.elements[0:self.settings["agents"]]
        self.agents = ["animal_"+str(idx) for idx in range(len(agents))]
        self.timestep = 0

        observations = {"animal_"+str(idx):{"observation":agent.collision_distance} for idx,agent in enumerate(agents)}

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos


    def render(self):
        """Renders the environment."""
        for idx,elem in enumerate(self.elements):
            elem.draw(self.screen)
        # Render on screen
        pygame.display.flip()

        self.clock.tick(60)  # limits FPS to 60

        #check if exit button is pressed in window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.running = False

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self.obs

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.acts
    

def otros():
    settings = {
        "agents":5,
        "obstacles":1,
        "food":1
    }
    env = Terrarium(settings)
    env.reset()
    while env.running:
        env.step({a:random.randint(0,5) for a in env.agents})
        env.render()

    pygame.quit()

if __name__ == "__main__":
    settings = {
        "agents":2,
        "obstacles":10,
        "food":5
    }
    env = Terrarium(settings)
    parallel_api_test(env, num_cycles=1_000_000)
