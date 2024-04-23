import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import terrarium.terrarium_v0 as Terrarium
from gymnasium.wrappers import RecordVideo
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


def create_my_env(env):
    env = RecordVideo(env(settings,"rgb_array"), video_folder="./save_videos",episode_trigger=lambda x: x % 2 == 0)
    # from gym_dog.envs.dog_env_2 import DogEnv2
    env = ParallelPettingZooEnv(env)
    return env

ray.init(ignore_reinit_error=True) #local_mode=True


select_env = "terrarium"
settings = {
        "agents":5,
        "obstacles":3,
        "food":5
    }

env_creator = lambda config: create_my_env(Terrarium.env)
register_env(select_env, env_creator)

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env=select_env)
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
