import terrarium.terrarium_v0 as Terrarium

if __name__ == "__main__":
    settings = {
        "agents":5,
        "obstacles":3,
        "food":5
    }
    env = Terrarium.env(settings)
    env.reset()
    while env.running:
        env.step({agent: env.action_space(agent).sample() for agent in env.agents})
        env.render()


def test():
    settings = {
        "agents":3,
        "obstacles":1,
        "food":5
    }
    env = Terrarium(settings)
    parallel_api_test(env, num_cycles=1_000_000)