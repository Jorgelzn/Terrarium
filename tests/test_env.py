from terrarium import terrarium_v0

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = terrarium_v0.parallel_env(x=2, y=2)
    parallel_api_test(env, num_cycles=1_000_000)
