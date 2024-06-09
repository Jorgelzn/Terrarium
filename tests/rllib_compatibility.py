import ray
from ray import tune
import os
from ray.rllib.algorithms.ppo import PPOConfig
from terrarium import terrarium_v0
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env


def env_creator(args):
    env = terrarium_v0.parallel_env()
    #env = ss.dtype_v0(env, "float32")
    #env = ss.resize_v1(env, x_size=84, y_size=84)
    return env


if __name__ == '__main__':
    ray.init()

    env_name = "terrarium_v0"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))


    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
