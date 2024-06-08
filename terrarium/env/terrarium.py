import functools

import gymnasium
import pygame
from gymnasium.spaces import Discrete
import random
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from terrarium.env.src import constants as const
from terrarium.env.src.Camera import Camera
from terrarium.env.src.entity import Entity

import numpy as np


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
        "render_modes": ["human"],
        "name": "terrarium",
        "render_fps": const.FPS
    }

    def __init__(self, render_mode=None, num_agents=4, voxels=20):
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
        self.agents_list = []
        self.action_masks = np.ones((num_agents, self.action_space(None).n))
        # Game Status
        self.frames = 0
        self.render_mode = render_mode
        self.screen = None
        self.camera = None
        self.voxels = voxels
        self.world_size = self.voxels * const.BLOCK_SIZE
        self.grid = []
        for y in range(0, self.world_size, const.BLOCK_SIZE):
            self.grid.append([])
            for x in range(0, self.world_size, const.BLOCK_SIZE):
                self.grid[-1].append(pygame.Rect(x, y, const.BLOCK_SIZE, const.BLOCK_SIZE))

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

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
                agent.do_action(actions["agent_" + str(idx)])
            self.check_actions()

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {agent: {} for agent in self.agents}

        terminations = {agent: False for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        self.time_steps += 1
        env_truncation = self.time_steps >= const.MAX_STEPS or self.screen is None
        truncations = {agent: env_truncation for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = {agent: {
            "observation": (),
            "action_mask": self.action_masks[idx]
        } for idx, agent in enumerate(self.agents)}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.spawn_agents()
        self.check_actions()
        self.time_steps = 0
        observations = {agent: {
            "observation": (),
            "action_mask": self.action_masks[idx]
        } for idx, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def spawn_agents(self):

        for _ in range(len(self.agents)):
            x = random.randrange(0, self.voxels)
            y = random.randrange(0, self.voxels)
            while self.occupied(x, y):
                x = random.randrange(0, self.voxels)
                y = random.randrange(0, self.voxels)

            self.agents_list.append(Entity(x, y))

    def occupied(self, x, y):
        for agent in self.agents_list:
            if x == agent.x and y == agent.y:
                return True
        return False

    def check_actions(self):
        for idx, agent in enumerate(self.agents_list):
            if agent.y - 1 == 0 or self.occupied(agent.x, agent.y - 1):
                self.action_masks[idx][0]=0
            else:
                self.action_masks[idx][0]=1
            if agent.y + 1 == self.voxels or self.occupied(agent.x, agent.y + 1):
                self.action_masks[idx][1]=0
            else:
                self.action_masks[idx][1] = 1
            if agent.x - 1 == 0 or self.occupied(agent.x - 1, agent.y):
                self.action_masks[idx][2]=0
            else:
                self.action_masks[idx][2] = 1
            if agent.x + 1 == self.voxels or self.occupied(agent.x + 1, agent.y):
                self.action_masks[idx][3]=0
            else:
                self.action_masks[idx][3] = 1

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)

    def draw(self):
        self.screen.fill((100, 100, 100))

        pygame.draw.rect(self.screen, (255, 100, 100), self.camera.apply(self.grid[0][0]))
        pygame.draw.rect(self.screen, (255, 100, 100), self.camera.apply(self.grid[0][-1]))
        pygame.draw.rect(self.screen, (255, 100, 100), self.camera.apply(self.grid[-1][0]))
        pygame.draw.rect(self.screen, (255, 100, 100), self.camera.apply(self.grid[-1][-1]))
        #pygame.draw.rect(self.screen, (100, 255, 100),self.camera.apply(self.grid[len(self.grid) // 2][len(self.grid[0]) // 2]))

        for agent in self.agents_list:
            pygame.draw.rect(self.screen, (100, 100, 255), self.camera.apply(self.grid[agent.y][agent.x]))

        for row in range(len(self.grid)):
            for colum, voxel in enumerate(self.grid[row]):
                pygame.draw.rect(self.screen, (255, 255, 255), self.camera.apply(voxel), 1)

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

        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    [const.SCREEN_WIDTH, const.SCREEN_HEIGHT]
                )
                self.camera = Camera(self.world_size)
                pygame.display.set_caption("Terrarium")
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.MOUSEBUTTONDOWN:
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
