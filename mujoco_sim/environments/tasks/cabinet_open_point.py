"""
idea of this env is to do sim2real on RGB(-D) observations
for learning closed-loop planar pushing of various items towards a target location (w/constant material)
kind of like procgen generalization but for manipulation.

and to test continuous vs discrete vs implicit (spatial actions)

and handheld camera vs static camera?

requires:
- a robot
- a workspace enlarging and self-collision avoiding EEF for dealing with UR3e constraints! (e.g Z shaped that can rotate around)
- an arena that is varied but should include environments similar to the real-world table
- random objects to push around
- an admittance controller to make the whole thing safe?

- if robot pushes something out of its workspace, the episode terminates with a penalty
    so don't babysit it by allowing only actions that would keep the object in the workspace..
- varying number of objects in the scene etc.
"""

import dataclasses
from typing import List, Tuple

import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from dm_env import specs

from mujoco_sim.entities.arenas import EmptyRobotArena
from mujoco_sim.entities.camera import Camera, CameraConfig
from mujoco_sim.entities.eef.cylinder import CylinderEEF
from mujoco_sim.entities.props.google_block import GoogleBlockProp
from mujoco_sim.entities.robots.robot import UR5e
from mujoco_sim.entities.utils import build_mocap, write_xml
from mujoco_sim.environments.tasks.base import TaskConfig
from mujoco_sim.entities.props.cabinet import HingeCabinet

DENSE_NEG_DISTANCE_REWARD = "dense_negative_distance_reward"


STATE_OBS = "state_observations"

REWARD_TYPES = (DENSE_NEG_DISTANCE_REWARD)
OBSERVATION_TYPES = (STATE_OBS)

TOP_DOWN_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])



@dataclasses.dataclass
class CabinetOpenPointConfig(TaskConfig):
    reward_type: str = DENSE_NEG_DISTANCE_REWARD
    observation_type: str = STATE_OBS
    max_step_size: float = 0.02
    physics_timestep: float = 0.002  # MJC default =0.002 (500Hz)
    control_timestep: float = 0.1
    max_control_steps_per_episode: int = 200
    goal_distance_threshold: float = 0.02  # task solved if dst(point,goal) < threshold


    def __post_init__(self):
        assert self.observation_type in OBSERVATION_TYPES
        assert self.reward_type in REWARD_TYPES


class CabinetOpenPoint(composer.Task):
    def __init__(self, config: CabinetOpenPointConfig) -> None:
        self.config = CabinetOpenPointConfig() if config is None else config

        self._arena = EmptyRobotArena(1)

        self.cabinet = HingeCabinet()
        self._arena.attach(self.cabinet)
        # creat mocap and attach to the handle.

        self.mocap = build_mocap(self._arena.mjcf_model, "control-mocap")
        self._arena.mjcf_model.equality.add(
            "weld",
            name="mocap_to_hanlde_weld",
            body1=self.cabinet.handle.mjcf_model.full_identifier,
            body2=self.mocap.full_identifier,


        )



        # set timesteps
        self.physics_timestep = self.config.physics_timestep
        self.control_timestep = self.config.control_timestep
        self.objects = []

        # create additional observables / Sensors
        self.goal_position_observable = observable.Generic(lambda physics: physics.bind(self.target).pos[:2])
        self._task_observables = {
            "target_position": self.goal_position_observable,
        }
        self._configure_observables()

    def _configure_observables(self):
        if self.config.observation_type == STATE_OBS:
            # TODO: enable hinge position observable

            pass

    def initialize_episode(self, physics, random_state):
        cabinet_position = np.array([0.0, -0.2, 0.0])
        self.cabinet.set_pose(physics, position = cabinet_position)
        physics.bind(self.mocap).pos =physics.named.data.site_xpos[self.cabinet.handle.grasp_site.full_identifier]

    @property
    def root_entity(self):
        return self._arena

    def before_step(self, physics, action, random_state):
        if action is None:
            return
        assert action.shape == (3,)
       

        # set mocap position


    def get_reward(self, physics):
        # return cabinet joint angle
        pass
    def action_spec(self, physics):
        return specs.BoundedArray(shape=(3,), dtype=float, minimum=-self.config.max_step_size,
                                  maximum=self.config.max_step_size)

    def get_discount(self, physics):
        return 0.0 if self.is_task_accomplished(physics) else 1.0

    def should_terminate_episode(self, physics) -> bool:
        accomplished = self.is_task_accomplished(physics)
        time_limit = physics.time() >= self.config.max_control_steps_per_episode * self.control_timestep

        return accomplished or time_limit

    def _get_object_distances_to_target(self, physics) -> List[float]:
        distances = []
        for object in self.objects:
            distance = np.linalg.norm(object.get_position(physics)[:2] - physics.bind(self.target).pos[:2])
            distances.append(distance)
        return distances

    def is_task_accomplished(self, physics) -> bool:
        #TODO: check if cabinet is open
        return False

    @property
    def task_observables(self):
        return {name: obs for (name, obs) in self._task_observables.items() if obs.enabled}


def create_random_policy(environment: composer.Environment):
    spec = environment.action_spec()
    environment.observation_spec()

    def random_policy(time_step):
        return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

    return random_policy


if __name__ == "__main__":
    from dm_control import viewer
    from dm_control.composer import Environment

    task = CabinetOpenPoint(CabinetOpenPointConfig())
    environment = Environment(task, strip_singleton_obs_buffer_dim=True)
    timestep = environment.reset()
    import matplotlib.pyplot as plt

    # plt.show()
    print(environment.action_spec())
    print(environment.observation_spec())
    # print(environment.step(None))
    write_xml(task._arena.mjcf_model)

    # import matplotlib.pyplot as plt
    # plt.imsave("test.png", timestep.observation["Camera/rgb_image"])

    viewer.launch(environment, policy=create_random_policy(environment))
