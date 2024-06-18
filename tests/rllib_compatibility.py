from terrarium import terrarium_v0
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import ray
from ray import tune

def env_creator(args):
    env = terrarium_v0.parallel_env(render_mode="human")
    #env = ss.dtype_v0(env, "float32")
    #env = ss.resize_v1(env, x_size=84, y_size=84)
    return env


if __name__ == '__main__':
    ray.init()

    env_name = "terrarium_v0"
    #ray.rllib.utils.check_env(ParallelPettingZooEnv(env_creator(0)))
    tune.register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    config = {
        "env": env_name,
        "num_workers": 4,
        "framework": "torch",
    }

    tune.run("PPO",config=config)
