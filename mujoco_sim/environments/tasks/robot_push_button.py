"""


"""

from __future__ import annotations

import numpy as np
from dm_control import composer, mjcf
from dm_env import specs

from mujoco_sim.entities.arenas import EmptyRobotArena
from mujoco_sim.entities.camera import Camera, CameraConfig
from mujoco_sim.entities.eef.gripper import Robotiq2f85
from mujoco_sim.entities.props.switch import Switch
from mujoco_sim.entities.robots.robot import UR5e
from mujoco_sim.environments.tasks.spaces import EuclideanSpace

TOP_DOWN_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


class RobotPushButtonTask(composer.Task):
    SPARSE_REWARD = "sparse_reward"

    STATE_OBS = "state_observations"
    VISUAL_OBS = "visual_observations"

    ABS_EEF_ACTION = "absolute_eef_action"
    ABS_JOINT_ACTION = "absolute_joint_action"

    REWARD_TYPES = SPARSE_REWARD
    OBSERVATION_TYPES = (STATE_OBS, VISUAL_OBS)
    ACTION_TYPES = (ABS_EEF_ACTION, ABS_JOINT_ACTION)

    MAX_STEP_SIZE: float = 0.05
    # TIMESTEP IS MAIN DRIVER OF SIMULATION SPEED..
    # HIGHER STEPS START TO RESULT IN UNSTABLE PHYSICS
    PHYSICS_TIMESTEP: float = 0.005  # MJC DEFAULT =0.002 (500HZ)
    CONTROL_TIMESTEP: float = 0.1
    MAX_CONTROL_STEPS_PER_EPISODE: int = 100

    GOAL_DISTANCE_THRESHOLD: float = 0.05  # TASK SOLVED IF DST(POINT,GOAL) < THRESHOLD
    TARGET_RADIUS = 0.03  # RADIUS OF THE TARGET SITE

    def __init__(
        self,
        reward_type: str = SPARSE_REWARD,
        observation_type: str = VISUAL_OBS,
        action_type: str = ABS_JOINT_ACTION,
        image_resolution: int = 96,
        scene_camera_position: np.ndarray = np.array([0.0, -1.7, 0.7]),
        scene_camera_orientation: np.ndarray = np.array([-0.7, -0.35, 0, 0.0]),
        use_wrist_camera: bool = True,
        wrist_camera_position: np.ndarray = np.array([0.0, 0.05, 0]),
        wrist_camera_orientation: np.ndarray = np.array([0.0, 0.0, 0.999, 0.04]),
        button_disturbances: bool = False,
    ) -> None:
        super().__init__()

        self.reward_type = reward_type
        self.observation_type = observation_type
        self.action_type = action_type
        self.image_resolution = image_resolution
        self.button_disturbances = button_disturbances

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
        self.robot_workspace = EuclideanSpace((-0.2, 0.2), (-0.6, -0.3), (0.02, 0.3))
        self.robot_spawn_space = EuclideanSpace((-0.2, 0.2), (-0.6, -0.3), (0.02, 0.3))
        self.target_spawn_space = EuclideanSpace((-0.2, 0.2), (-0.6, -0.3), (0.0, 0.1))

        # for debugging camera views etc: add workspace to scene
        # self.workspace_geom = self.robot_workspace.create_visualization_site(self._arena.mjcf_model.worldbody,"robot-workspace")

        # add Camera to scene
        self.scene_camera_config = CameraConfig(scene_camera_position, scene_camera_orientation, 70,image_height=self.image_resolution,image_width=self.image_resolution)
        self.camera = Camera(self.scene_camera_config)
        self._arena.attach(self.camera)

        self.use_wrist_camera = use_wrist_camera
        if use_wrist_camera:
            self.wrist_camera_config = CameraConfig(wrist_camera_position, wrist_camera_orientation, 42,image_height=self.image_resolution,image_width=self.image_resolution)
            self.wrist_camera = Camera(self.wrist_camera_config)
            self.wrist_camera.observables.rgb_image.name = "wrist_camera_rgb_image"
            self.robot.attach(self.wrist_camera, self.robot.flange)

        # create additional observables / Sensors
        self._task_observables = {}
        #TODO: use this to add robot state to observation space?


        self._configure_observables()

        # set timesteps
        # has to happen here as the _arena has to be available.
        self.physics_timestep = self.PHYSICS_TIMESTEP
        self.control_timestep = self.CONTROL_TIMESTEP

        self.robot_end_position = np.array([-0.3, -0.2, 0.3])  # end position of the robot once the switch is activated

    def _configure_observables(self):
        if self.action_type == RobotPushButtonTask.ABS_EEF_ACTION:
            self.robot.observables.tcp_position.enabled = True
        else:
            self.robot.observables.joint_configuration.enabled = True
        # TODO: add gripper state.

        if self.observation_type == RobotPushButtonTask.STATE_OBS:
            self.switch.observables.position.enabled = True
            self.switch.observables.active.enabled = True

        elif self.observation_type == RobotPushButtonTask.VISUAL_OBS:
            self.camera.observables.rgb_image.enabled = True
            if self.use_wrist_camera:
                self.wrist_camera.observables.rgb_image.enabled = True

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        robot_initial_pose = self.robot_spawn_space.sample(random_state)
        robot_initial_pose = np.concatenate([robot_initial_pose, TOP_DOWN_QUATERNION])
        self.robot.set_tcp_pose(physics, robot_initial_pose)

        # switch  position
        switch_position = self.target_spawn_space.sample(random_state)
        self.switch.set_pose(physics, position=switch_position)

        # print(f"target position: {target_position}")
        # print(self.goal_position_observable(physics))

    @property
    def root_entity(self):
        return self._arena

    def before_step(self, physics, action, random_state):
        if self.action_type == RobotPushButtonTask.ABS_EEF_ACTION:
            assert action.shape == (4,)
            griper_target = action[3]
            robot_position = action[:3]

            self.gripper.move(physics, griper_target)
            self.robot.servoL(physics, np.concatenate([robot_position, TOP_DOWN_QUATERNION]), self.control_timestep)

        elif self.action_type == RobotPushButtonTask.ABS_JOINT_ACTION:
            assert action.shape == (7,)
            gripper_target = action[6]
            joint_configuration = action[:6]
            self.gripper.move(physics, gripper_target)
            self.robot.servoJ(physics, joint_configuration, self.control_timestep)

    def after_step(self, physics, random_state):
        # if the button is active, with some probability make it inactive
        if self.button_disturbances:
            if (
                self.switch.is_active and not self.switch._is_pressed and random_state.rand() < 0.01
            ):  # (0.99)**30 = 0.74 probability to reach end pose before disturbance.
                self.switch.deactivate(physics)

    def get_reward(self, physics):

        if self.reward_type == RobotPushButtonTask.SPARSE_REWARD:
            return self.is_goal_reached(physics) * 1.0

    def action_spec(self, physics):
        del physics
        # bound = np.array([self.MAX_STEP_SIZE, self.MAX_STEP_SIZE])
        # normalized action space, rescaled in before_step
        if self.action_type == RobotPushButtonTask.ABS_EEF_ACTION:
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
        elif self.action_type == RobotPushButtonTask.ABS_JOINT_ACTION:
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

    def is_goal_reached(self, physics) -> bool:
        return (
            self.switch.is_active
            and np.linalg.norm(self.robot.get_tcp_pose(physics)[:3] - self.robot_end_position)
            < self.GOAL_DISTANCE_THRESHOLD
        )

    def should_terminate_episode(self, physics):
        return self.is_goal_reached(physics)

    def get_discount(self, physics):
        if self.should_terminate_episode(physics):
            return 0.0
        else:
            return 1.0

    def create_random_policy(self):
        physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        spec = self.action_spec(physics)

        def random_policy(time_step):
            # return np.array([0.01, 0])
            return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

        return random_policy

    def create_demonstration_policy(self, environment):  # noqa C901
        def demonstration_policy(time_step: composer.TimeStep):
            physics = environment.physics
            # get the current physics state
            # get the current robot pose
            robot_position = self.robot.get_tcp_pose(physics).copy()
            robot_position = robot_position[:3]

            # get the current target pose
            switch_position = self.switch.get_position(physics).copy()
            target_position = switch_position
            # target_position[2] += 0.
            is_switch_active = self.switch.is_active

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
            # print(f"phase: {phase}")
            # print(f"target position: {target_position}")
            if phase == 1:
                # move towards the target, first move up to avoid collisions
                if robot_position[2] < target_position[2] + 0.02:
                    action = robot_position
                    action[2] = target_position[2] + 0.05
                    # print(f"moving up to {action}")
                else:
                    action = target_position
                    action[2] = target_position[2] + 0.05
                    # print("moving towards")

            if phase == 2:
                # move down to the target
                action = target_position

            if phase == 3:
                # move to the end pose
                action = self.robot_end_position.copy()
                if np.linalg.norm(switch_position[:2] - robot_position[:2]) < 0.05:
                    action[2] = target_position[2] + 0.1  # avoid touching button upon moving to end position

            # calculate the action to reach the target
            difference = action - robot_position[:3]

            MAX_SPEED = 0.5
            if np.max(np.abs(difference)) > MAX_SPEED * environment.control_timestep():
                difference = difference * MAX_SPEED / np.max(np.abs(difference)) * environment.control_timestep()
            action = robot_position[:3] + difference

            # if needed, convert action to joint configuration
            if self.action_type == RobotPushButtonTask.ABS_JOINT_ACTION:
                tcp_pose = np.concatenate([action, TOP_DOWN_QUATERNION])
                current_joint_positions = self.robot.get_joint_positions(physics)
                action = self.robot.get_joint_positions_from_tcp_pose(tcp_pose, current_joint_positions)

            # add gripper, which is always closed
            action = np.concatenate([action, [0.0]])
            return action

        return demonstration_policy


if __name__ == "__main__":
    from dm_control import viewer
    from dm_control.composer import Environment

    task = RobotPushButtonTask(
        reward_type=RobotPushButtonTask.SPARSE_REWARD,
        observation_type=RobotPushButtonTask.VISUAL_OBS,
        action_type=RobotPushButtonTask.ABS_JOINT_ACTION,
        button_disturbances=True,
        image_resolution=256,
    )

    # dump task xml

    # mjcf.export_with_assets(task._arena.mjcf_model, ".")

    environment = Environment(
        task,
        strip_singleton_obs_buffer_dim=True,
        time_limit=task.MAX_CONTROL_STEPS_PER_EPISODE * task.CONTROL_TIMESTEP,
    )
    timestep = environment.reset()

    print(timestep.observation)

    # plt.imshow(timestep.observation["Camera/rgb_image"])
    # plt.show()
    print(environment.action_spec())
    print(environment.observation_spec())
    img = task.camera.get_rgb_image(environment.physics)

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.imsave("test.png", img)

    #viewer.launch(environment, policy=task.create_demonstration_policy(environment))
