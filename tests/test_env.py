from terrarium import terrarium_v0

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = terrarium_v0.parallel_env(render_mode="human")
    #parallel_api_test(env, num_cycles=1_000_000)

    observations, infos = env.reset()

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        #print(actions)
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()
