"""
This task is a simple planar push task
 where the robot has to push a (number of) blocks to a target location.
Planar in the sense that the robot only controls the xy position of the end effector.
"""
from __future__ import annotations

import dataclasses
from typing import List

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
from mujoco_sim.environments.tasks.base import RobotTask, TaskConfig
from mujoco_sim.environments.tasks.spaces import EuclideanSpace

TOP_DOWN_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


@dataclasses.dataclass
class RobotPushConfig(TaskConfig):
    # add these macros in the class to make it easier to use them
    # without having to import them separately
    SPARSE_REWARD = "sparse_reward"
    DENSE_NEG_DISTANCE_REWARD = "dense_negative_distance_reward"

    STATE_OBS = "state_observations"
    VISUAL_OBS = "visual_observations"

    REWARD_TYPES = (SPARSE_REWARD, DENSE_NEG_DISTANCE_REWARD)
    OBSERVATION_TYPES = (STATE_OBS, VISUAL_OBS)

    TOP_DOWN_CAMERA_CONFIG = CameraConfig(np.array([0.0, -0.5, 2.4]), np.array([1.0, 0.0, 0.0, 0.0]), 30)
    FRONT_TILTED_CAMERA_CONFIG = CameraConfig(np.array([0.0, -1.1, 0.5]), np.array([-0.7, -0.35, 0, 0.0]), 70)

    # actual config
    reward_type: str = None
    observation_type: str = None
    max_step_size: float = 0.05
    # timestep is main driver of simulation speed..
    # higher steps start to result in unstable physics
    physics_timestep: float = 0.005  # MJC default =0.002 (500Hz)
    control_timestep: float = 0.1
    max_control_steps_per_episode: int = 500
    goal_distance_threshold: float = 0.02  # task solved if dst(point,goal) < threshold
    image_resolution: int = 64

    # coef for the additional reward term that encourages the robot
    # to touch the objects, only used with dense rewards.
    nearest_object_reward_coefficient: float = 0.1
    target_radius = 0.05  # radius of the target site
    n_objects: int = 5  # number of objects to push

    cameraconfig: CameraConfig = None

    def __post_init__(self):
        # set default values if not set
        # https://stackoverflow.com/questions/56665298/how-to-apply-default-value-to-python-dataclass-field-when-none-was-passed
        self.reward_type = self.reward_type or RobotPushConfig.DENSE_NEG_DISTANCE_REWARD
        self.observation_type = self.observation_type or RobotPushConfig.STATE_OBS
        self.cameraconfig = self.cameraconfig or RobotPushConfig.FRONT_TILTED_CAMERA_CONFIG

        assert self.observation_type in RobotPushConfig.OBSERVATION_TYPES
        assert self.reward_type in RobotPushConfig.REWARD_TYPES


class RobotPushTask(RobotTask):
    def __init__(self, config: RobotPushConfig) -> None:
        super().__init__(config)
        self.config: RobotPushConfig  # noqa

        # create arena, robot and EEF
        self._arena = EmptyRobotArena(3)
        self.robot = UR5e()
        self.cylinderEEF = CylinderEEF()
        self.robot.attach_end_effector(self.cylinderEEF)

        # get the xml of the robot
        write_xml(self.robot.mjcf_model)
        self._arena.attach(self.robot, self._arena.robot_attachment_site)

        # creat target
        self.target = self._arena.mjcf_model.worldbody.add(
            "site",
            name="target",
            type="cylinder",
            rgba=[1, 1, 1, 1.0],
            size=[config.target_radius, 0.001],
            pos=[0.0, -0.5, 0.001],
        )

        # create robot workspace and all the spawn spaces
        self.robot_workspace = EuclideanSpace((-0.2, 0.2), (-0.6, -0.3), (0.03, 0.015))
        self.robot_spawn_space = EuclideanSpace((-0.2, 0.2), (-0.6, -0.3), (0.02, 0.02))
        self.object_spawn_space = EuclideanSpace((-0.15, 0.15), (-0.55, -0.35), (0.05, 0.2))
        self.target_spawn_space = EuclideanSpace((-0.15, 0.15), (-0.55, -0.35), (0.001, 0.005))

        # for debugging camera views etc: add workspace to scene
        # self.workspace_geom = self.robot_workspace.create_visualization_site(self._arena.mjcf_model.worldbody,"robot-workspace")
        # add Camera to scene
        camera_config = self.config.cameraconfig
        camera_config.image_width = camera_config.image_height = self.config.image_resolution
        self.camera = Camera(camera_config)
        self._arena.attach(self.camera)

        # create dummy objects to initialize the observables
        self.objects = []
        self._create_objects()

        # create additional observables / Sensors
        self.goal_position_observable = observable.Generic(lambda physics: physics.bind(self.target).pos[:2])
        self.block_position_observable = observable.Generic(self.get_object_positions)

        self._task_observables = {
            "target_position": self.goal_position_observable,
            "block_positions": self.block_position_observable,
        }
        self._configure_observables()

        # set timesteps
        # has to happen here as the _arena has to be available.
        self.physics_timestep = self.config.physics_timestep
        self.control_timestep = self.config.control_timestep

    def _configure_observables(self):
        if self.config.observation_type == RobotPushConfig.STATE_OBS:
            self.goal_position_observable.enabled = True
            self.block_position_observable.enabled = True
            self.robot.observables.tcp_position.enabled = True

        elif self.config.observation_type == RobotPushConfig.VISUAL_OBS:
            self.camera.observables.rgb_image.enabled = True
            self.robot.observables.tcp_position.enabled = True

    def initialize_episode_mjcf(self, random_state):
        for object in self.objects:
            object.detach()
        self._create_objects()

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        robot_initial_pose = self.robot_spawn_space.sample()
        self.robot.set_tcp_pose(physics, np.concatenate([robot_initial_pose, TOP_DOWN_QUATERNION]))

        target_position = self.target_spawn_space.sample()
        physics.bind(self.target).pos = target_position

        self.randomize_object_position(physics)

        # give objects time to drop onto the table
        for _ in range(150):
            physics.step()

    def _create_objects(self):
        self.objects = [GoogleBlockProp.sample_random_object() for _ in range(self.config.n_objects)]
        self.object_joints = []
        for object in self.objects:
            self.object_joints.append(self._arena.attach(object).add("freejoint"))

    def randomize_object_position(self, physics):
        colliding = True
        while colliding:
            for object_joint in self.object_joints:
                physics.named.data.qpos[object_joint.full_identifier][:3] = self.object_spawn_space.sample()
                physics.named.data.qpos[object_joint.full_identifier][3:] = np.array([1, 0, 0, 0])
            physics.forward()  # forward to set new poses and update collision detection
            colliding = physics.data.ncon > 0

    def get_object_positions(self, physics):
        return np.array([object.get_position(physics)[:2] for object in self.objects]).flatten()

    @property
    def root_entity(self):
        return self._arena

    def before_step(self, physics, action, random_state):
        """

        Args:
            action (_type_): [-1,1] x action_dim
        """

        super().before_step(physics, action, random_state)
        if action is None:
            return
        assert action.shape == (2,)

        target_position = np.zeros((3,))
        target_position[:2] = action
        target_position[2] = 0.02  # keep z position constant
        # target_position = self.robot_workspace.clip_to_space(target_position)
        self.robot.servoL(physics, np.concatenate([target_position, TOP_DOWN_QUATERNION]), self.control_timestep)

    def get_reward(self, physics):
        if self.config.reward_type == RobotPushConfig.SPARSE_REWARD:
            distances = self._get_object_distances_to_target(physics)
            return sum([distance < self.config.target_radius for distance in distances])
        else:
            reward = -sum(self._get_object_distances_to_target(physics)) / self.config.n_objects

            # get distance between robot and objects to encourage robot to move (exploration)
            distance_to_nearest_object = min(
                [
                    np.linalg.norm(self.robot.get_tcp_pose(physics)[:2] - object.get_position(physics)[:2])
                    for object in self.objects
                ]
            )
            reward -= self.config.nearest_object_reward_coefficient * distance_to_nearest_object
            # scale rewards to reduce effort for critic to
            # learn the initial q-values of all states
            return reward * 0.1

    def _get_object_distances_to_target(self, physics) -> List[float]:
        distances = []
        for object in self.objects:
            distance = np.linalg.norm(object.get_position(physics)[:2] - physics.bind(self.target).pos[:2])
            distances.append(distance)
        return distances

    def action_spec(self, physics):
        del physics
        # bound = np.array([self.config.max_step_size, self.config.max_step_size])
        # normalized action space, rescaled in before_step
        bound = np.array([1.0, 1.0])
        return specs.BoundedArray(shape=(2,), dtype=np.float32, minimum=-bound, maximum=bound)

    def is_task_accomplished(self, physics) -> bool:
        distances = self._get_object_distances_to_target(physics)
        return all([distance < self.config.target_radius for distance in distances])

    def should_terminate_episode(self, physics):
        return self.is_task_accomplished(physics) or self.current_step >= self.config.max_control_steps_per_episode


def create_random_policy(environment: composer.Environment):
    spec = environment.action_spec()
    environment.observation_spec()

    def random_policy(time_step):
        # return np.array([0.01, 0])
        print(time_step.reward)
        return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

    return random_policy


import sys
import termios
import tty

import click


def getch():
    """
    Get a single character from the terminal, wait for 0.01s before releasing the keyboard if no key is pressed.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def print_keyboard_options():
    click.secho("\n       Keyboard Controls:", fg="yellow")
    click.secho("=====================================", fg="yellow")
    print("W / A /S / D : Move BASE")
    print("U / J / H / K : Move LIFT & ARM")
    print("N / M : Open & Close GRIPPER")
    print("Q : Stop")
    click.secho("=====================================", fg="yellow")


def create_demonstration_policy(environment: composer.Environment):
    def demonstration_policy(time_step):

        print_keyboard_options()
        key = getch().lower()

        # use arrows to control the robot

        if key == "j":
            eef_action = np.array([0.0, 0.1])
        elif key == "l":
            eef_action = np.array([0.0, -0.1])
        elif key == "i":
            eef_action = np.array([0.1, 0.0])
        elif key == "k":
            eef_action = np.array([-0.1, 0.0])
        else:
            eef_action = np.array([0.0, 0.0])
        return eef_action

    return demonstration_policy


if __name__ == "__main__":
    from dm_control.composer import Environment

    task = RobotPushTask(
        RobotPushConfig(observation_type=RobotPushConfig.STATE_OBS, nearest_object_reward_coefficient=0.1, n_objects=2)
    )

    environment = Environment(task, strip_singleton_obs_buffer_dim=True)
    timestep = environment.reset()

    # plt.imshow(timestep.observation["Camera/rgb_image"])
    # plt.show()
    print(environment.action_spec())
    print(environment.observation_spec())
    print(timestep.observation)

    # viewer.launch(environment, policy=create_demonstration_policy(environment))

    environment.reset()
    done = False
    import cv2

    window = cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    while not done:
        img = environment.physics.render(camera_id=0)
        cv2.imshow("image", img)
        key = cv2.waitKey(0)
        action = environment.task.robot.get_tcp_pose(environment.physics)[:2]

        if key == ord("j"):
            action += np.array([0.0, 0.05])
        elif key == ord("l"):
            action += np.array([0.0, -0.05])
        elif key == ord("i"):
            action += np.array([0.05, 0.0])
        elif key == ord("k"):
            action += np.array([-0.05, 0.0])
        elif key == ord("q"):
            break
        timestep = environment.step(action)
        done = timestep.last()

    # mouse position on the viewer
