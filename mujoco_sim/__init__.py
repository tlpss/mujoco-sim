import pathlib

_LOGGING_DIR = pathlib.Path(__file__).parents[1] / "logging"


# register environments here for 'lerobot' train script


from functools import partial

import gymnasium
from dm_control.composer import Environment as DMCEnvironment

from mujoco_sim.environments.dmc2gym import DMCEnvironmentAdapter
from mujoco_sim.environments.tasks.point_reach import PointMassReachTask
from mujoco_sim.environments.tasks.robot_push_button import RobotPushButtonTask
import numpy as np 

class LerobotGymWrapper(gymnasium.Wrapper):
    def __init__(self, env, image_key_mapping, state_keys):
        super().__init__(env)

        self.image_key_mapping = image_key_mapping
        self.state_keys = state_keys

    @property
    def observation_space(self):
        agent_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(sum([self.env.observation_space[key].shape[0] for key in self.state_keys]),))
        pix_spaces = {}
        for key, new_key in self.image_key_mapping.items():
            if key in self.env.observation_space.keys():
                pix_spaces[new_key] = self.env.observation_space[key]

        return gymnasium.spaces.Dict({
            "agent_pos": agent_space,
            "pixels": gymnasium.spaces.Dict(pix_spaces)
        })

    def transform_observation(self, observation):
        new_observation = {}
        new_observation["agent_pos"] = np.concatenate([observation[key] for key in self.state_keys],axis=0)
        new_observation["pixels"] = {}
        for key, new_key in self.image_key_mapping.items():
            if key in observation:
                new_observation["pixels"][new_key] = observation[key]
        return new_observation
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.transform_observation(observation), reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        observation,info = self.env.reset(**kwargs)
        return self.transform_observation(observation),info
    

    
    @property
    def action_space(self):
        return self.env.action_space
    

    
 


def make_point_mass_reach_env(task_class, max_steps, **kwargs):
    task = task_class(**kwargs)
    env = DMCEnvironment(task, strip_singleton_obs_buffer_dim=True, time_limit=max_steps * task.CONTROL_TIMESTEP)
    gym_env = DMCEnvironmentAdapter(env, flatten_observation_space=False)
    return gym_env

def make_lerobot_env(task_class, max_steps, **kwargs):
    env = task_class(**kwargs)
    env = DMCEnvironment(env, strip_singleton_obs_buffer_dim=True, time_limit=max_steps * env.CONTROL_TIMESTEP)
    env = DMCEnvironmentAdapter(env, flatten_observation_space=False, render_dims=(256, 256),render_camera_id=0)
    gym_env = LerobotGymWrapper(env,{"Camera/rgb_image": "scene", "ur5e/Camera/rgb_image": "wrist"},["ur5e/joint_configuration"])
    return gym_env

gymnasium.register(
    id="mujoco_sim/point_mass_reach-v0",
    entry_point=partial(make_point_mass_reach_env, PointMassReachTask, max_steps=50),
    kwargs={"observation_type": "visual_observations", "image_resolution": 64},
    max_episode_steps=50,
)

gymnasium.register(
    id="mujoco_sim/robot_push_button_visual-v0",
    entry_point=partial(make_lerobot_env, RobotPushButtonTask, max_steps=150),
    kwargs={
    },
    max_episode_steps=150,
)


if __name__ == "__main__":

    # list all envs
    #print(gymnasium.registry.keys())

    env = gymnasium.make("mujoco_sim/robot_push_button_visual-v0",image_resolution=256,use_wrist_camera=False)

    print(env.observation_space)
    obs,_ = env.reset()
    img = env.render()
    print(obs)
    import matplotlib.pyplot as plt
    plt.imsave("test.png",img)