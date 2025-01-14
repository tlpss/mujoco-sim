import time

import gym


def benchmark_step_time(env: gym.Env, n_steps=2000):
    done = True
    start = time.time()
    for i in range(n_steps):
        if done:
            obs = env.reset()  # noqa
            done = False
        else:
            obs, reward, done, info = env.step(env.action_space.sample())  # noqa
    stop = time.time()

    return {"per_step": (stop - start) / n_steps, "total": (stop - start)}


def cProfile(func, *args, **kwargs):
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    print(func(*args, **kwargs))
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(50)


if __name__ == "__main__":
    from dm_control.composer import Environment

    from mujoco_sim.environments.dmc2gym import DMCEnvironmentAdapter
    from mujoco_sim.environments.tasks.robot_planar_push import RobotPushConfig, RobotPushTask

    config = RobotPushConfig(observation_type="visual_observations")
    dmc_env = Environment(RobotPushTask(config=config), strip_singleton_obs_buffer_dim=True)
    gym_env = DMCEnvironmentAdapter(dmc_env, flatten_observation_space=False)

    print(cProfile(benchmark_step_time, gym_env, 1000))
