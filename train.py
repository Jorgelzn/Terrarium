
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray import air, tune
import terrarium.terrarium_v0 as Terrarium
from gymnasium.wrappers import RecordVideo

if __name__ == "__main__":
    settings = {
        "agents":5,
        "obstacles":3,
        "food":5
    }
    #env_creator(settings,False)
    ray.init(num_cpus=4)
    trigger = lambda t: t % 10 == 0
    env_name = "terrarium"
    register_env(env_name, lambda config: ParallelPettingZooEnv(RecordVideo(Terrarium.env(settings,"rgb_array"), video_folder="./save_videos", episode_trigger=trigger, disable_logger=True)))

    # Example config switching on rendering.
    config = (
        PPOConfig()
        # Also try common gym envs like: "CartPole-v1" or "Pendulum-v1".
        .environment(env=env_name,
            env_config={"corridor_length": 10, "max_steps": 100},
        )
        .rollouts(num_envs_per_worker=2, num_rollout_workers=1)
        .evaluation(
            # Evaluate once per training iteration.
            evaluation_interval=1,
            # Run evaluation on (at least) two episodes
            evaluation_duration=2,
            # ... using one evaluation worker (setting this to 0 will cause
            # evaluation to run on the local evaluation worker, blocking
            # training until evaluation is done).
            evaluation_num_workers=1,
            # Special evaluation config. Keys specified here will override
            # the same keys in the main config, but only for evaluation.
            evaluation_config=PPOConfig.overrides(
                # Render the env while evaluating.
                # Note that this will always only render the 1st RolloutWorker's
                # env and only the 1st sub-env in a vectorized env.
                render_env=True,
            ),
        )
    )

    stop = {
        "training_iteration": 100,
        "timesteps_total": 100,
        "episode_reward_mean": 5,
    }

    tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop),
    ).fit()