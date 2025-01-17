import pathlib

_LOGGING_DIR = pathlib.Path(__file__).parents[1] / "logging"


# register environments here for 'lerobot' train script


from functools import partial

import gymnasium
from dm_control.composer import Environment as DMCEnvironment

from mujoco_sim.environments.dmc2gym import DMCEnvironmentAdapter
from mujoco_sim.environments.tasks.point_reach import PointMassReachTask
from mujoco_sim.environments.tasks.robot_push_button import RobotPushButtonTask


def make_point_mass_reach_env(task_class, max_steps, **kwargs):
    task = task_class(**kwargs)
    env = DMCEnvironment(task, strip_singleton_obs_buffer_dim=True, time_limit=max_steps * task.CONTROL_TIMESTEP)
    gym_env = DMCEnvironmentAdapter(env, flatten_observation_space=False)
    return gym_env


gymnasium.register(
    id="mujoco_sim/point_mass_reach-v0",
    entry_point=partial(make_point_mass_reach_env, PointMassReachTask, max_steps=50),
    kwargs={"observation_type": "visual_observations", "image_resolution": 64},
)

gymnasium.register(
    id="mujoco_sim/robot_push_button_visual-v0",
    entry_point=partial(make_point_mass_reach_env, RobotPushButtonTask, max_steps=100),
    kwargs={
        "observation_type": RobotPushButtonTask.VISUAL_OBS,
        "image_resolution": 96,
        "action_type": RobotPushButtonTask.ABS_JOINT_ACTION,
    },
)


if __name__ == "__main__":

    # list all envs
    print(gymnasium.registry.keys())
    import cv2

    env = gymnasium.make("mujoco_sim/point_mass_reach-v0")

    done = False
    obs = env.reset()
    action_policy = env.unwrapped._env.task.create_random_policy()
    n = 0
    while not done:
        obs, reward, term, trunc, info = env.step(action_policy(obs))
        n += 1
        print(n)
        done = term or trunc
        print(done)
        img = env.render()
        cv2.imshow("img", img)
        cv2.waitKey(1)
