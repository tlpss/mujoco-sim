"""
This task is a simple Reach task where the robot has to reach a target position with its end effector.

Observations are either proprioception and image or proprioception and target position

Rewards are either sparse or dense negative distance reward

Action space is either target TCP poses or target joint configurations

The task is solved if the distance between the robot's end effector and the target is less than a threshold

"""

from __future__ import annotations

import dataclasses

import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from dm_env import specs

from mujoco_sim.entities.arenas import EmptyRobotArena
from mujoco_sim.entities.camera import Camera, CameraConfig
from mujoco_sim.entities.eef.gripper import Robotiq2f85
from mujoco_sim.entities.robots.robot import UR5e
from mujoco_sim.environments.tasks.base import TaskConfig
from mujoco_sim.environments.tasks.spaces import EuclideanSpace

TOP_DOWN_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


@dataclasses.dataclass
class RobotReachConfig(TaskConfig):
    # add these macros in the class to make it easier to use them
    # without having to import them separately
    SPARSE_REWARD = "sparse_reward"
    DENSE_NEG_DISTANCE_REWARD = "dense_negative_distance_reward"

    STATE_OBS = "state_observations"
    VISUAL_OBS = "visual_observations"

    REL_EEF_ACTION = "relative_eef_action"
    ABS_EEF_ACTION = "absolute_eef_action"
    REL_JOIN_ACTION = "relative_joint_action"
    ABS_JOIN_ACTION = "absolute_joint_action"

    REWARD_TYPES = (SPARSE_REWARD, DENSE_NEG_DISTANCE_REWARD)
    OBSERVATION_TYPES = (STATE_OBS, VISUAL_OBS)
    ACTION_TYPES = (REL_EEF_ACTION, ABS_EEF_ACTION, REL_JOIN_ACTION, ABS_JOIN_ACTION)

    FRONT_TILTED_CAMERA_CONFIG = CameraConfig(np.array([0.0, -1.1, 0.5]), np.array([-0.7, -0.35, 0, 0.0]), 70)

    # actual config
    reward_type: str = None
    observation_type: str = None
    action_type: str = None

    max_step_size: float = 0.05
    # timestep is main driver of simulation speed..
    # higher steps start to result in unstable physics
    physics_timestep: float = 0.005  # MJC default =0.002 (500Hz)
    control_timestep: float = 0.1
    max_control_steps_per_episode: int = 100
    image_resolution: int = 96

    goal_distance_threshold: float = 0.02  # task solved if dst(point,goal) < threshold
    target_radius = 0.03  # radius of the target site

    cameraconfig: CameraConfig = None

    def __post_init__(self):
        # set default values if not set
        # https://stackoverflow.com/questions/56665298/how-to-apply-default-value-to-python-dataclass-field-when-none-was-passed
        self.reward_type = self.reward_type or RobotReachConfig.DENSE_NEG_DISTANCE_REWARD
        self.observation_type = self.observation_type or RobotReachConfig.STATE_OBS
        self.action_type = self.action_type or RobotReachConfig.ABS_EEF_ACTION
        self.cameraconfig = self.cameraconfig or RobotReachConfig.FRONT_TILTED_CAMERA_CONFIG

        assert self.observation_type in RobotReachConfig.OBSERVATION_TYPES
        assert self.reward_type in RobotReachConfig.REWARD_TYPES
        assert self.action_type in RobotReachConfig.ACTION_TYPES


class RobotReachTask(composer.Task):
    def __init__(self, config: RobotReachConfig) -> None:
        super().__init__()
        self.config: RobotReachConfig = config

        # create arena, robot and EEF
        self._arena = EmptyRobotArena(3)
        self.robot = UR5e()
        self.gripper = Robotiq2f85()
        self.robot.attach_end_effector(self.gripper)
        self._arena.attach(self.robot, self._arena.robot_attachment_site)

        # creat target
        self.target = self._arena.mjcf_model.worldbody.add(
            "site",
            name="target",
            type="sphere",
            rgba=[1, 1, 1, 1.0],
            size=[config.target_radius],
            pos=[0.0, -0.5, 0.001],
        )

        # create robot workspace and all the spawn spaces
        self.robot_workspace = EuclideanSpace((-0.1, 0.1), (-0.6, -0.4), (0.02, 0.2))
        self.robot_spawn_space = EuclideanSpace((-0.1, 0.1), (-0.6, -0.4), (0.02, 0.2))
        self.target_spawn_space = EuclideanSpace((-0.1, 0.1), (-0.6, -0.4), (0.02, 0.2))

        # for debugging camera views etc: add workspace to scene
        # self.workspace_geom = self.robot_workspace.create_visualization_site(self._arena.mjcf_model.worldbody,"robot-workspace")

        # add Camera to scene
        camera_config = self.config.cameraconfig
        camera_config.image_width = camera_config.image_height = self.config.image_resolution
        self.camera = Camera(camera_config)
        self._arena.attach(self.camera)

        # create additional observables / Sensors
        self.goal_position_observable = observable.Generic(lambda physics: physics.bind(self.target).pos[:3])

        self._task_observables = {
            "target_position": self.goal_position_observable,
        }
        self._configure_observables()

        # set timesteps
        # has to happen here as the _arena has to be available.
        self.physics_timestep = self.config.physics_timestep
        self.control_timestep = self.config.control_timestep

    def _configure_observables(self):
        if self.config.observation_type == RobotReachConfig.STATE_OBS:
            self.goal_position_observable.enabled = True
            self.robot.observables.tcp_position.enabled = True

        elif self.config.observation_type == RobotReachConfig.VISUAL_OBS:
            self.robot.observables.tcp_position.enabled = True
            self.camera.observables.rgb_image.enabled = True

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        robot_initial_pose = self.robot_spawn_space.sample()
        robot_initial_pose = np.concatenate([robot_initial_pose, TOP_DOWN_QUATERNION])
        # ŧarget position
        self.robot.set_tcp_pose(physics, robot_initial_pose)
        target_position = self.target_spawn_space.sample()
        physics.bind(self.target).pos = target_position

        # print(f"target position: {target_position}")
        # print(self.goal_position_observable(physics))

    @property
    def root_entity(self):
        return self._arena

    def before_step(self, physics, action, random_state):
        """

        Args:
            action (_type_): [-1,1] x action_dim
        """
        if action is None:
            return
        assert action.shape == (3,)

        self.robot.servoL(physics, np.concatenate([action, TOP_DOWN_QUATERNION]), self.control_timestep)

    def _robot_distance_to_target(self, physics):
        return np.linalg.norm(self.robot.get_tcp_pose(physics)[:3] - physics.bind(self.target).xpos[:3])

    def get_reward(self, physics):

        if self.config.reward_type == RobotReachConfig.SPARSE_REWARD:
            return self.is_task_accomplished(physics)

        elif self.config.reward_type == RobotReachConfig.DENSE_NEG_DISTANCE_REWARD:
            distance = self._robot_distance_to_target(physics)
            return -distance

    def action_spec(self, physics):
        del physics
        # bound = np.array([self.config.max_step_size, self.config.max_step_size])
        # normalized action space, rescaled in before_step
        if self.config.action_type == RobotReachConfig.ABS_EEF_ACTION:
            return specs.BoundedArray(
                shape=(3,),
                dtype=np.float64,
                minimum=[
                    self.robot_workspace.x_range[0],
                    self.robot_workspace.y_range[0],
                    self.robot_workspace.z_range[0],
                ],
                maximum=[
                    self.robot_workspace.x_range[1],
                    self.robot_workspace.y_range[1],
                    self.robot_workspace.z_range[1],
                ],
            )

        if self.config.action_type == RobotReachConfig.ABS_JOIN_ACTION:
            return specs.BoundedArray()

    def is_task_accomplished(self, physics) -> bool:
        return self._robot_distance_to_target(physics) < self.config.goal_distance_threshold


def create_random_policy(environment: composer.Environment):
    spec = environment.action_spec()
    environment.observation_spec()

    def random_policy(time_step):
        # return np.array([0.01, 0])
        print(time_step.reward)
        return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

    return random_policy


def create_demonstration_policy(environment: composer.Environment, noise_level=0.0):
    def demonstration_policy(time_step: composer.TimeStep):

        assert isinstance(environment.task, RobotReachTask)
        assert (
            environment.task.config.action_type == RobotReachConfig.ABS_EEF_ACTION
        ), "only implemented demonstrator for EEF actions for now"
        # get the current physics state
        physics = environment.physics
        # get the current robot pose
        robot_pose = environment.task.robot.get_tcp_pose(physics).copy()

        # get the current target pose
        target_pose = physics.bind(environment.task.target).xpos.copy()

        print("robot pose", robot_pose)
        print("target pose", target_pose)

        # calculate the action to reach the target
        difference = target_pose - robot_pose[:3]

        # add noise
        difference += np.random.normal(0, noise_level, size=3)

        # move at most 0.5m/s
        if np.max(np.abs(difference)) > 0.5:
            difference = difference * 0.5 / np.max(np.abs(difference)) * environment.control_timestep()
        action = robot_pose[:3] + difference
        # action = np.array([0.2,-0.2,0.2])
        return action

    return demonstration_policy


if __name__ == "__main__":
    from dm_control import viewer
    from dm_control.composer import Environment

    task = RobotReachTask(RobotReachConfig(observation_type=RobotReachConfig.STATE_OBS))

    # dump task xml

    # mjcf.export_with_assets(task._arena.mjcf_model, ".")

    environment = Environment(task, strip_singleton_obs_buffer_dim=True)
    timestep = environment.reset()

    print(timestep.observation)

    # plt.imshow(timestep.observation["Camera/rgb_image"])
    # plt.show()
    print(environment.action_spec())
    print(environment.observation_spec())
    img = task.camera.get_rgb_image(environment.physics)

    # plt.imshow(img)
    # plt.show()

    viewer.launch(environment, policy=create_demonstration_policy(environment))
