"""


"""
from __future__ import annotations

import dataclasses

import numpy as np
from dm_control import composer
from dm_env import specs

from mujoco_sim.entities.arenas import EmptyRobotArena
from mujoco_sim.entities.camera import Camera, CameraConfig
from mujoco_sim.entities.eef.gripper import Robotiq2f85
from mujoco_sim.entities.props.switch import Switch
from mujoco_sim.entities.robots.robot import UR5e
from mujoco_sim.environments.tasks.base import TaskConfig
from mujoco_sim.environments.tasks.spaces import EuclideanSpace

TOP_DOWN_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


@dataclasses.dataclass
class RobotPushButtonConfig(TaskConfig):
    # add these macros in the class to make it easier to use them
    # without having to import them separately
    SPARSE_REWARD = "sparse_reward"

    STATE_OBS = "state_observations"
    VISUAL_OBS = "visual_observations"

    ABS_EEF_ACTION = "absolute_eef_action"
    ABS_JOIN_ACTION = "absolute_joint_action"

    REWARD_TYPES = SPARSE_REWARD
    OBSERVATION_TYPES = (STATE_OBS, VISUAL_OBS)
    ACTION_TYPES = (ABS_EEF_ACTION, ABS_JOIN_ACTION)

    FRONT_TILTED_CAMERA_CONFIG = CameraConfig(np.array([0.0, -1.1, 0.5]), np.array([-0.7, -0.35, 0, 0.0]), 70)
    WRIST_CAMERA_CONFIG = CameraConfig(
        position=np.array([0.0, 0.05, 0]), orientation=np.array([0.0, 0.0, 0.999, 0.04]), fov=42, name="WristCamera"
    )

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

    goal_distance_threshold: float = 0.05  # task solved if dst(point,goal) < threshold
    target_radius = 0.03  # radius of the target site

    scene_camera_config: CameraConfig = None
    wrist_camera_config: CameraConfig = None

    def __post_init__(self):
        # set default values if not set
        # https://stackoverflow.com/questions/56665298/how-to-apply-default-value-to-python-dataclass-field-when-none-was-passed
        self.reward_type = self.reward_type or RobotPushButtonConfig.SPARSE_REWARD
        self.observation_type = self.observation_type or RobotPushButtonConfig.STATE_OBS
        self.action_type = self.action_type or RobotPushButtonConfig.ABS_EEF_ACTION
        self.scene_camera_config = self.scene_camera_config or RobotPushButtonConfig.FRONT_TILTED_CAMERA_CONFIG
        self.wrist_camera_config = self.wrist_camera_config or RobotPushButtonConfig.WRIST_CAMERA_CONFIG

        assert self.observation_type in RobotPushButtonConfig.OBSERVATION_TYPES
        assert self.reward_type in RobotPushButtonConfig.REWARD_TYPES
        assert self.action_type in RobotPushButtonConfig.ACTION_TYPES


class RobotPushButtonTask(composer.Task):
    def __init__(self, config: RobotPushButtonConfig) -> None:
        super().__init__()
        self.config: RobotPushButtonConfig = config

        # create arena, robot and EEF
        self._arena = EmptyRobotArena(3)
        self.robot = UR5e()
        self.gripper = Robotiq2f85()
        self.robot.attach_end_effector(self.gripper)
        self._arena.attach(self.robot, self._arena.robot_attachment_site)

        self.switch = Switch()

        # attach switch to world
        self._arena.attach(self.switch)

        # create robot workspace and all the spawn spaces
        self.robot_workspace = EuclideanSpace((-0.1, 0.1), (-0.6, -0.4), (0.02, 0.2))
        self.robot_spawn_space = EuclideanSpace((-0.1, 0.1), (-0.6, -0.4), (0.02, 0.2))
        self.target_spawn_space = EuclideanSpace((-0.1, 0.1), (-0.6, -0.4), (0.0, 0.0))

        # for debugging camera views etc: add workspace to scene
        # self.workspace_geom = self.robot_workspace.create_visualization_site(self._arena.mjcf_model.worldbody,"robot-workspace")

        # add Camera to scene
        camera_config = self.config.scene_camera_config
        self.camera = Camera(camera_config)
        self._arena.attach(self.camera)

        self.wrist_camera = Camera(self.config.wrist_camera_config)
        self.wrist_camera.observables.rgb_image.name = "wrist_camera_rgb_image"
        self.robot.attach(self.wrist_camera, self.robot.flange)

        # create additional observables / Sensors

        self._task_observables = {}
        self._configure_observables()

        # set timesteps
        # has to happen here as the _arena has to be available.
        self.physics_timestep = self.config.physics_timestep
        self.control_timestep = self.config.control_timestep

        self.robot_end_position = np.array([-0.5, -0.5, 0.3])  # end position of the robot once the switch is activated

    def _configure_observables(self):
        if self.config.observation_type == RobotPushButtonConfig.STATE_OBS:
            # self.switch.observables.position.enabled = True
            # self.switch.observables.active.enabled = True
            self.robot.observables.tcp_position.enabled = True

        elif self.config.observation_type == RobotPushButtonConfig.VISUAL_OBS:
            self.robot.observables.tcp_position.enabled = True
            self.camera.observables.rgb_image.enabled = True
            self.wrist_camera.observables.rgb_image.enabled = True

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        robot_initial_pose = self.robot_spawn_space.sample()
        robot_initial_pose = np.concatenate([robot_initial_pose, TOP_DOWN_QUATERNION])
        self.robot.set_tcp_pose(physics, robot_initial_pose)

        # switch  position
        switch_position = self.target_spawn_space.sample()
        self.switch.set_pose(physics, position=switch_position)

        # print(f"target position: {target_position}")
        # print(self.goal_position_observable(physics))

    @property
    def root_entity(self):
        return self._arena

    def before_step(self, physics, action, random_state):
        if self.config.action_type == RobotPushButtonConfig.ABS_EEF_ACTION:
            assert action.shape == (4,)
            griper_target = action[3]
            robot_position = action[:3]

            self.gripper.move(physics, griper_target)
            self.robot.servoL(physics, np.concatenate([robot_position, TOP_DOWN_QUATERNION]), self.control_timestep)

        elif self.config.action_type == RobotPushButtonConfig.ABS_JOIN_ACTION:
            assert action.shape == (7,)
            gripper_target = action[6]
            joint_configuration = action[:6]
            self.gripper.move(physics, gripper_target)
            self.robot.servoJ(physics, joint_configuration, self.control_timestep)

    def get_reward(self, physics):

        if self.config.reward_type == RobotPushButtonConfig.SPARSE_REWARD:
            return self.is_task_accomplished(physics) * 1.0

    def action_spec(self, physics):
        del physics
        # bound = np.array([self.config.max_step_size, self.config.max_step_size])
        # normalized action space, rescaled in before_step
        if self.config.action_type == RobotPushButtonConfig.ABS_EEF_ACTION:
            return specs.BoundedArray(
                shape=(4,),
                dtype=np.float64,
                minimum=[
                    self.robot_workspace.x_range[0],
                    self.robot_workspace.y_range[0],
                    self.robot_workspace.z_range[0],
                    0.0,
                ],
                maximum=[
                    self.robot_workspace.x_range[1],
                    self.robot_workspace.y_range[1],
                    self.robot_workspace.z_range[1],
                    self.gripper.open_distance,
                ],
            )
        elif self.config.action_type == RobotPushButtonConfig.ABS_JOIN_ACTION:
            return specs.BoundedArray(
                shape=(7,),
                dtype=np.float64,
                minimum=[
                    -3.14,
                ]
                * 6
                + [0.0],
                maximum=[3.14] * 6 + [0.085],
            )

    def is_task_accomplished(self, physics) -> bool:
        print(np.linalg.norm(self.robot.get_tcp_pose(physics)[:3] - self.robot_end_position))
        return (
            self.switch.is_active
            and np.linalg.norm(self.robot.get_tcp_pose(physics)[:3] - self.robot_end_position)
            < self.config.goal_distance_threshold
        )

    def should_terminate_episode(self, physics):
        return self.is_task_accomplished(physics)


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

        assert isinstance(environment.task, RobotPushButtonTask)

        # get the current physics state
        physics = environment.physics
        # get the current robot pose
        robot_position = environment.task.robot.get_tcp_pose(physics).copy()
        robot_position = robot_position[:3]

        # get the current target pose
        switch_position = environment.task.switch.get_position(physics).copy()
        target_position = switch_position
        target_position[2] += 0.01
        is_switch_active = environment.task.switch.is_active

        # if robot is not above the switch and switch is not active, this is phase 1
        # if robot is above that pose and switch is not active, this is phase 2
        # if switch is active, this is phase 3

        if is_switch_active:
            phase = 3
        elif (
            robot_position[2] > target_position[2]
            and np.linalg.norm(robot_position[:2] - target_position[:2]) < 0.01
            and not is_switch_active
        ):
            phase = 2
        else:
            phase = 1
        print(f"phase: {phase}")
        print(f"target position: {target_position}")
        if phase == 1:
            # move towards the target, first move up to avoid collisions
            if robot_position[2] < target_position[2] + 0.02:
                action = robot_position
                action[2] = target_position[2] + 0.05
                print(f"moving up to {action}")
            else:
                action = target_position
                action[2] = target_position[2] + 0.05
                print("moving towards")

        if phase == 2:
            # move down to the target
            action = target_position

        if phase == 3:
            # move to the end pose
            action = environment.task.robot_end_position

        # calculate the action to reach the target
        difference = action - robot_position[:3]

        MAX_SPEED = 0.5
        if np.max(np.abs(difference)) > MAX_SPEED * environment.control_timestep():
            difference = difference * MAX_SPEED / np.max(np.abs(difference)) * environment.control_timestep()
        action = robot_position[:3] + difference
        # #action = np.array([0.2,-0.2,0.2])

        # if needed, convert to joint space

        if environment.task.config.action_type == RobotPushButtonConfig.ABS_JOIN_ACTION:
            tcp_pose = np.concatenate([action, TOP_DOWN_QUATERNION])
            current_joint_positions = environment.task.robot.get_joint_positions(physics)
            action = environment.task.robot.get_joint_positions_from_tcp_pose(tcp_pose, current_joint_positions)
            print(action)

        # add gripper, which is always closed
        action = np.concatenate([action, [0.0]])
        return action

    return demonstration_policy


if __name__ == "__main__":
    from dm_control import viewer
    from dm_control.composer import Environment

    task = RobotPushButtonTask(
        RobotPushButtonConfig(
            observation_type=RobotPushButtonConfig.VISUAL_OBS, action_type=RobotPushButtonConfig.ABS_JOIN_ACTION
        )
    )

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
