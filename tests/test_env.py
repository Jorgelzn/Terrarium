from terrarium import terrarium_v0

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = terrarium_v0.parallel_env(voxels=2,num_agents=4)
    #parallel_api_test(env, num_cycles=1_000_000)


    limit = 1000
    timer = limit

    observations, infos = env.reset()
    env.render()

    while env.screen is not None:
        env.render()
        timer = timer-env.clock.get_time()

        if timer < 0:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            timer = limit

    env.close()
