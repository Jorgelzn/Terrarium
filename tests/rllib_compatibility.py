from terrarium import terrarium_v0
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import ray


def env_creator(args):
    env = terrarium_v0.parallel_env(x=2, y=2)
    #env = ss.dtype_v0(env, "float32")
    #env = ss.resize_v1(env, x_size=84, y_size=84)
    return env


if __name__ == '__main__':
    ray.init()

    env_name = "terrarium_v0"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
