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
from mujoco_sim.entities.utils import write_xml
from mujoco_sim.environments.tasks.base import TaskConfig

SPARSE_REWARD = "sparse_reward"
DENSE_POTENTIAL_REWARD = "dense_potential_reward"
DENSE_NEG_DISTANCE_REWARD = "dense_negative_distance_reward"
DENSE_BIASED_NEG_DISTANCE_REWARD = "dense_biased_negative_distance_reward"

STATE_OBS = "state_observations"
VISUAL_OBS = "visual_observations"

REWARD_TYPES = (SPARSE_REWARD, DENSE_POTENTIAL_REWARD, DENSE_NEG_DISTANCE_REWARD, DENSE_BIASED_NEG_DISTANCE_REWARD)
OBSERVATION_TYPES = (STATE_OBS, VISUAL_OBS)

TOP_DOWN_CAMERA_CONFIG = CameraConfig(np.array([0.0, -0.5, 2.4]), np.array([1.0, 0.0, 0.0, 0.0]), 30)

TOP_DOWN_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


class RobotPositionWorkspace:
    def __init__(
        self, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float]
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def clip_to_workspace(self, pose: np.ndarray) -> np.ndarray:
        for i, axis in enumerate([self.x_range, self.y_range, self.z_range]):
            pose[i] = np.clip(pose[i], axis[0], axis[1])
        return pose

    def is_in_workspace(self, pose: np.ndarray) -> bool:
        if np.isclose(pose, self.clip_to_workspace(pose)).all():
            return True
        return False

    def sample(self) -> np.ndarray:
        return np.array(
            [np.random.uniform(*self.x_range), np.random.uniform(*self.y_range), np.random.uniform(*self.z_range)]
        )


@dataclasses.dataclass
class RobotPushConfig(TaskConfig):
    reward_type: str = DENSE_NEG_DISTANCE_REWARD
    observation_type: str = STATE_OBS
    max_step_size: float = 0.02
    physics_timestep: float = 0.002  # MJC default =0.002 (500Hz)
    control_timestep: float = 0.1
    max_control_steps_per_episode: int = 200
    goal_distance_threshold: float = 0.02  # task solved if dst(point,goal) < threshold
    image_resolution: int = 128

    target_radius = 0.1
    n_objects: int = 5

    def __post_init__(self):
        assert self.observation_type in OBSERVATION_TYPES
        assert self.reward_type in REWARD_TYPES


class RobotPushTask(composer.Task):
    def __init__(self, config: RobotPushConfig) -> None:
        self.config = RobotPushConfig() if config is None else config

        self._arena = EmptyRobotArena(3)
        self.robot = UR5e()
        self.robot_site = self._arena.mjcf_model.worldbody.add("site", name="robot_site", pos=[0.0, 0.0, 0])
        self.cylinderEEF = CylinderEEF()
        self.robot.attach_end_effector(self.cylinderEEF)
        # TODO: bring this site to the arena and standardize
        self._arena.attach(self.robot, self.robot_site)

        # creat target
        self.target = self._arena.mjcf_model.worldbody.add(
            "site",
            name="target",
            type="cylinder",
            rgba=[1, 1, 1, 1.0],
            size=[config.target_radius, 0.001],
            pos=[0.0, -0.5, 0.001],
        )

        self.robot_workspace = RobotPositionWorkspace((-0.5, 0.5), (-0.7, -0.3), (0.03, 0.03))
        self.object_spawn_space = RobotPositionWorkspace((-0.4, 0.4), (-0.6, -0.4), (0.05, 0.2))
        self.target_spawn_space = RobotPositionWorkspace((-0.3, 0.3), (-0.6, -0.4), (0.001, 0.001))

        # add Camera to scene
        top_down_config = TOP_DOWN_CAMERA_CONFIG
        top_down_config.image_width = top_down_config.image_height = self.config.image_resolution
        self.camera = Camera(top_down_config)
        self._arena.attach(self.camera)

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
            self.goal_position_observable.enabled = True
            self.robot.observables.tcp_position.enabled = True
            # TODO: add object positions

        elif self.config.observation_type == VISUAL_OBS:
            self.camera.observables.rgb_image.enabled = True

    def initialize_episode_mjcf(self, random_state):
        for object in self.objects:
            object.detach()
        self._create_objects()

    def initialize_episode(self, physics, random_state):
        robot_initial_pose = self.robot_workspace.sample()
        self.robot.set_tcp_pose(physics, np.concatenate([robot_initial_pose, TOP_DOWN_QUATERNION]))

        target_position = self.target_spawn_space.sample()
        physics.bind(self.target).pos = target_position

        self.randomize_object_position(physics)

        # self.block.set_pose(physics, np.concatenate([self.robot_workspace.sample(), TOP_DOWN_QUATERNION]))

    def _create_objects(self):
        self.objects = [GoogleBlockProp.sample_random_object() for _ in range(self.config.n_objects)]
        self.object_joints = []
        for object in self.objects:
            self.object_joints.append(self._arena.attach(object).add("freejoint"))
        # randomize locations
        # and repeat until everything is collision free

    def randomize_object_position(self, physics):

        colliding = True
        while colliding:
            for object_joint in self.object_joints:
                physics.named.data.qpos[object_joint.full_identifier][:3] = self.object_spawn_space.sample()
                physics.named.data.qpos[object_joint.full_identifier][3:] = np.array([1, 0, 0, 0])
            colliding = False  # physics.data.ncon > 0

    @property
    def root_entity(self):
        return self._arena

    def before_step(self, physics, action, random_state):
        if action is None:
            return
        assert action.shape == (2,)
        current_position = self.robot.get_tcp_pose(physics)[:3]
        print("-")
        print(current_position)
        target_position = np.copy(current_position)
        target_position[:2] += action
        target_position = self.robot_workspace.clip_to_workspace(target_position)
        print(target_position)
        self.robot.servoL(physics, np.concatenate([target_position, TOP_DOWN_QUATERNION]), self.control_timestep)

    def get_reward(self, physics):
        if self.config.reward_type == SPARSE_REWARD:
            distances = self._get_object_distances_to_target(physics)
            return sum([distance < self.config.target_radius for distance in distances])
        else:
            # normalize by max distance to get a reward between 0 and 1
            max_distances = self.config.n_objects * 1.0  # raw approximation
            return (max_distances - sum(self._get_object_distances_to_target(physics))) / max_distances

    def action_spec(self, physics):
        del physics
        bound = np.array([self.config.max_step_size, self.config.max_step_size])
        return specs.BoundedArray(shape=(2,), dtype=np.float32, minimum=-bound, maximum=bound)

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
        distances = self._get_object_distances_to_target(physics)
        return all([distance < self.config.target_radius for distance in distances])

    @property
    def task_observables(self):
        return {name: obs for (name, obs) in self._task_observables.items() if obs.enabled}


def create_random_policy(environment: composer.Environment):
    spec = environment.action_spec()
    environment.observation_spec()

    def random_policy(time_step):
        return np.array([0.01, 0])
        return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

    return random_policy


if __name__ == "__main__":
    from dm_control import viewer
    from dm_control.composer import Environment

    task = RobotPushTask(RobotPushConfig(observation_type=VISUAL_OBS))
    environment = Environment(task, strip_singleton_obs_buffer_dim=True)
    timestep = environment.reset()
    import matplotlib.pyplot as plt

    plt.imshow(timestep.observation["Camera/rgb_image"])
    # plt.show()
    print(environment.action_spec())
    print(environment.observation_spec())
    # print(environment.step(None))
    write_xml(task._arena.mjcf_model)
    img = task.camera.get_rgb_image(environment.physics)

    # import matplotlib.pyplot as plt
    # plt.imsave("test.png", timestep.observation["Camera/rgb_image"])

    viewer.launch(environment, policy=create_random_policy(environment))
