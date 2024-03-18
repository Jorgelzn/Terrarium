import random
from copy import copy

import numpy as np

from pettingzoo import ParallelEnv

from pettingzoo.test import parallel_api_test

import pygame
from .engine import Food,Agent,Obstacle

class env(ParallelEnv):
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

        self.timestep = 0
        self.possible_agents = ["animal_"+str(idx) for idx in range(settings["agents"])]
        self.screen = pygame.display.set_mode((1280, 500))
        self.clock = pygame.time.Clock()
        self.running = True
        self.friction = 0.1
        self.settings = settings

        self.elements = []
        for obj_def in self.settings:
            for obj_num in range(settings[obj_def]):
                created = False
                while not created:
                    pos = np.array([random.uniform(0,self.screen.get_width()), random.uniform(0,self.screen.get_height())])
                    if obj_def ==  "agents":
                        obj = Agent(pos,40,4,random.randint(0, 360),200,"black","animal_"+str(obj_num))
                    elif obj_def == "obstacles":
                        obj = Obstacle(pos, np.array([random.uniform(0,300),random.uniform(0,300)]), "black")
                    elif obj_def == "food":
                        obj = Food(pos,30,"green")
                    collisions = [elem for elem in self.elements if elem.collision_rect.colliderect(obj.collision_rect)]
                    if not collisions:
                            self.elements.append(obj)
                            created = True
        self.init_elements = copy(self.elements)
        self.agents_obs = self.elements[:self.settings["agents"]]
        self.elements = self.elements[self.settings["agents"]+1:]
        pygame.init()

    def step(self,actions):
        # clean screen
        self.screen.fill("white")
        for idx,agent in enumerate(self.agents_obs):
            collisions = [idx for idx,elem in enumerate(self.elements) if elem.collision_rect.colliderect(agent.collision_rect)]
            agents_collisions=[idx for idx,a in enumerate(self.agents_obs) if a.collision_rect.colliderect(agent.collision_rect) and a!=agent]
            if not collisions and not agents_collisions:
                agent.move(agent.actions[actions[agent.agent_id]],self.friction)
            else:
                bounce = True
                for c in collisions:
                    if type(self.elements[c]) is Food:
                        self.elements.pop(c)
                        bounce = False
                        break
                if bounce:
                    agent.dv = -agent.dv

            agent.update()

            for idx,vision in enumerate(agent.vision):
                collided = False
                for a in self.agents_obs:
                    if a!=agent:
                        for elem in self.elements:
                            line_collide_agent = a.collision_rect.clipline(agent.pos,vision)
                            line_collide = elem.collision_rect.clipline(agent.pos,vision) 
                            if line_collide or line_collide_agent:
                                if line_collide:
                                    agent.vision[idx]=np.array(line_collide[0])
                                elif line_collide_agent:
                                    agent.vision[idx]=np.array(line_collide_agent[0])
                                agent.vision_color[idx]="red"
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
            self.agents_obs = []
            self.running = False


        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100000:
            rewards = {a:0 for a in self.agents}
            truncations = {a: True for a in self.agents}
            self.agents = []
            self.agents_obs = []
        self.timestep += 1

        # Get observations
        observations = {a.agent_id:{"observation":agent.collision_distance} for a in self.agents_obs}


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
        self.agents_obs = self.elements[:self.settings["agents"]]
        self.elements = self.elements[self.settings["agents"]+1:]
        self.agents = [a.agent_id for a in self.agents_obs]
        self.timestep = 0

        observations = {agent.agent_id:{"observation":agent.collision_distance} for agent in self.agents_obs}

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos


    def render(self):
        """Renders the environment."""
        for elem in self.elements:
            elem.draw(self.screen)
        for agent in self.agents_obs:
            agent.draw(self.screen)
        # Render on screen
        pygame.display.flip()

        self.clock.tick(60)  # limits FPS to 60

        #check if exit button is pressed in window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self.agents_obs[self.agents.index(agent)].obs

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.agents_obs[self.agents.index(agent)].acts
    

