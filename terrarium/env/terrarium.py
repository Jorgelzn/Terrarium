import functools

import gymnasium
import pygame
from gymnasium.spaces import Discrete,Dict
import random
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from terrarium.env.src import constants as const
from terrarium.env.src.Camera import Camera
from terrarium.env.src.entity import Entity
from terrarium.env.src.terrain import Terrain
import numpy as np
import noise

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {
        "render_modes": ["human","rgb_array"],
        "name": "terrarium",
        "render_fps": const.FPS
    }

    def __init__(self, voxels, num_agents, render_mode="human",render_obs=False):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["agent_" + str(r) for r in range(num_agents)]
        # Game Status
        self.frames = 0
        self.render_mode = render_mode
        self.screen = None
        self.camera = None
        self.voxels = voxels
        self.world_size = self.voxels * const.BLOCK_SIZE

        ## TERRAIN ##
        self.terrain = Terrain(self.world_size,const.BLOCK_SIZE)

        ## SPRITES

        self.ground_agent_sprite = pygame.image.load("../terrarium/env/data/treecko.png")
        self.ground_agent_sprite = pygame.transform.scale(self.ground_agent_sprite, (const.BLOCK_SIZE, const.BLOCK_SIZE))

        self.water_agent_sprite = pygame.image.load("../terrarium/env/data/mudkip.png")
        self.water_agent_sprite = pygame.transform.scale(self.water_agent_sprite,(const.BLOCK_SIZE, const.BLOCK_SIZE))

        #RENDERING, revisar
        pygame.init()
        self.clock = pygame.time.Clock()
        self.render_obs = render_obs
        self.camera = Camera(self.world_size)
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                [const.SCREEN_WIDTH, const.SCREEN_HEIGHT]
            )
            pygame.display.set_caption("Terrarium")
        elif self.render_mode == "rgb_array":
            self.screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))




    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        else:
            for idx, agent in enumerate(self.agents_list):
                action = actions["agent_" + str(idx)]
                agent.do_action(action, self.terrain.agents)



        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {agent: 0 for agent in self.agents}

        terminations = {agent: False for agent in self.agents}

        if self.render_mode == "rgb_array":
            self.render()

        self.time_steps += 1
        env_truncation = self.time_steps >= const.MAX_STEPS or self.screen is None
        truncations = {agent: env_truncation for agent in self.agents}

        observations = {agent_id: self.locate_agent(agent_id).get_observation(self.terrain) for idx, agent_id in enumerate(self.agents)}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []
            #self.close()

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Returns the observations for each agent
        """

        self.agents = self.possible_agents[:]
        self.spawn_agents()

        self.time_steps = 0
        observations = {agent_id: self.locate_agent(agent_id).get_observation(self.terrain) for idx, agent_id in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def spawn_agents(self):
        self.agents_list = []
        for idx in range(len(self.agents)):
            x = random.randrange(0, self.voxels)
            y = random.randrange(0, self.voxels)
            while self.terrain.agents[y][x] != 0:
                x = random.randrange(0, self.voxels)
                y = random.randrange(0, self.voxels)

            if random.random() < 0.5:
                sprite = self.ground_agent_sprite
            else:
                sprite = self.water_agent_sprite

            self.agents_list.append(Entity("agent_"+str(idx),x, y, sprite))
            self.terrain.agents[y][x] = 1

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return  self.locate_agent(agent_id).observation_space


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)

    def draw(self):
        self.screen.fill((100, 100, 100))

        # DRAW TERRAIN
        for idx_y,y in enumerate(self.terrain.terrain_type):
            for idx_x,x in enumerate(y):
                for texture in self.terrain.terrain_textures[idx_y][idx_x]:
                    self.screen.blit(texture, self.camera.apply(self.terrain.draw_grid[idx_y][idx_x]))
                if x == 1:
                    #UPDATE WATER ANIMATION TEXTURE
                    self.terrain.terrain_textures[idx_y][idx_x][0] = self.terrain.water[self.terrain.water_anim]

        #WATER ANIMATION TIMER
        self.terrain.water_anim_timer += self.clock.get_time()
        if self.terrain.water_anim_timer >= 90:
            self.terrain.water_anim_timer = 0
            if self.terrain.water_anim == 7:
                self.terrain.water_anim = 0
            else:
                self.terrain.water_anim += 1

        # DRAW OBSERVATION SPACES
        for idx,agent in enumerate(self.agents_list):
            if self.render_obs:
                for ob_id_y in agent.obs_ids:
                    for ob_idx in ob_id_y:
                        if ob_idx[0]!=-1 and (ob_idx[0]!=agent.y or ob_idx[1]!=agent.x):
                            rect_surface = pygame.Surface((const.BLOCK_SIZE, const.BLOCK_SIZE), pygame.SRCALPHA, 32)
                            rect_surface.set_alpha(128)
                            rect_surface.fill((255,0,100))
                            self.screen.blit(rect_surface, self.camera.apply(self.terrain.draw_grid[ob_idx[0]][ob_idx[1]]))

        # DRAW AGENTS
        for idx,agent in enumerate(self.agents_list):
            self.screen.blit(agent.sprite, self.camera.apply(self.terrain.draw_grid[agent.y][agent.x]))

        # DRAW GRID
        #for row in range(len(self.draw_grid)):
        #    for colum, voxel in enumerate(self.draw_grid[row]):
        #        pygame.draw.rect(self.screen, (255, 255, 255), self.camera.apply(voxel), 1)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

                if self.world_size > const.SCREEN_WIDTH or self.world_size > const.SCREEN_HEIGHT:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left mouse button
                            self.camera.start_drag(event.pos)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:  # Left mouse button
                            self.camera.stop_drag()
                    elif event.type == pygame.MOUSEMOTION:
                        if self.camera.dragging:
                            self.camera.update_drag(event.pos)

        if self.screen is not None:
            self.draw()
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            if self.render_mode == "human":
                pygame.display.flip()
                self.clock.tick(self.metadata["render_fps"])
            return (
                np.transpose(observation, axes=(1, 0, 2))
                if self.render_mode == "rgb_array"
                else None
            )

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def locate_agent(self,agent_id):
        for agent in self.agents_list:
            if agent.id == agent_id:
                return agent
