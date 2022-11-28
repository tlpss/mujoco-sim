import time

import gym


def benchmark_step_time(env: gym.Env, n_steps=2000):
    done = True
    start = time.time()
    for _ in range(n_steps):
        if done:
            obs = env.reset()  # noqa
            done = False
        else:
            obs, reward, done, info = env.step(env.action_space.sample())  # noqa
    stop = time.time()

    return (stop - start) / n_steps


if __name__ == "__main__":
    from dm_control.composer import Environment

    from mujoco_sim.environments.dmc2gym import DMCWrapper
    from mujoco_sim.environments.tasks.point_reach import PointMassReachTask, PointReachConfig

    config = PointReachConfig(observation_type="visual_observations")
    dmc_env = Environment(PointMassReachTask(config=config), strip_singleton_obs_buffer_dim=True)
    gym_env = DMCWrapper(dmc_env, flatten_observation_space=False)

    print(benchmark_step_time(gym_env))
