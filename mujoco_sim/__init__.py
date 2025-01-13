import pathlib

_LOGGING_DIR = pathlib.Path(__file__).parents[1] / "logging"


# register environments here for 'lerobot' train script


import gymnasium
from dm_control.composer import Environment as DMCEnvironment

from mujoco_sim.environments.dmc2gym import DMCWrapper
from mujoco_sim.environments.tasks.point_reach import PointMassReachTask


def make_point_mass_reach_env(**kwargs):
    task = PointMassReachTask(**kwargs)
    env = DMCEnvironment(task, strip_singleton_obs_buffer_dim=True)
    gym_env = DMCWrapper(env, flatten_observation_space=False)
    return gym_env


gymnasium.register(
    id="mujoco_sim/point_mass_reach-v0",
    entry_point=make_point_mass_reach_env,
    # max_episode_steps=50,
    kwargs={"observation_type": "visual_observations", "image_resolution": 64},
)
