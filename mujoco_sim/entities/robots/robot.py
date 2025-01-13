from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable

from mujoco_sim.entities.eef.cylinder import EEF
from mujoco_sim.entities.robots.joint_trajectory import JointTrajectory, Waypoint
from mujoco_sim.entities.robots.SE3Container import SE3Container
from mujoco_sim.type_aliases import JOINT_CONFIGURATION_TYPE, POSE_TYPE


class IKSolver:
    def solve_ik(self, pose: POSE_TYPE, q_guess: JOINT_CONFIGURATION_TYPE) -> Optional[JOINT_CONFIGURATION_TYPE]:
        raise NotImplementedError("IK solver not implemented")


class AnalyticURIKsolver(IKSolver):
    def __init__(self, robot_type: str = "ur5e"):
        try:
            pass
        except ImportError:
            raise ImportError(
                "ur_analytic_ik is not installed. Please install it from https://github.com/Victorlouisdg/ur-analytic-ik"
            )
        self.robot_type = robot_type

    def solve_ik(self, pose: POSE_TYPE, q_guess: JOINT_CONFIGURATION_TYPE) -> Optional[JOINT_CONFIGURATION_TYPE]:
        if self.robot_type == "ur5e":
            from ur_analytic_ik import ur5e

            # convert pose to homogeneous matrix
            pose = SE3Container.from_quaternion_and_translation(pose[3:], pose[:3]).homogeneous_matrix
            q = ur5e.inverse_kinematics_closest(pose, *q_guess)
            return q


class Robot(composer.Entity):
    """
    position-controlled Robot base class
    """

    _MODEL_XML_PATH = None

    _BASE_BODY_NAME: str = None
    # site that defines the frame to attach EEF
    #  (and is used internally for IK as it is the last element in the robot kinematics tree)
    _FLANGE_SITE_NAME: str = None
    home_joint_positions: JOINT_CONFIGURATION_TYPE = None
    max_joint_speed: float = None

    def __init__(self, ik_solver: IKSolver) -> None:
        """
        ik_
        """

        # Tool Center Frame translation wrt to the FLANGE (IK etc is based on this frame)
        self.tcp_in_flange_pose: POSE_TYPE = np.array([0, 0, 0, 0, 0, 0, 1.0])
        self.joint_trajectory: Optional[JointTrajectory] = None
        self.ik_solver = ik_solver

        # used for visualization of the performances of the low-level controllers
        self.joint_position_history = deque(maxlen=10000)
        self.joint_target_history = deque(maxlen=10000)

        self._reset_targets()
        super().__init__()

    def _build(self):
        self._model = mjcf.from_path(self._MODEL_XML_PATH)
        self.joints = self._model.find_all("joint")
        self.actuators = self._model.find_all("actuator")
        self.dof = len(self.joints)
        self.base_element = self._model.find("body", self._BASE_BODY_NAME)
        self.flange: mjcf.Element = self._model.find("site", self._FLANGE_SITE_NAME)

    def attach_end_effector(self, end_effector: EEF):

        # expand keyframe of the robot arm to account for the new DoFs
        # code taken from https://github.com/google-deepmind/mujoco_menagerie/blob/main/FAQ.md#how-do-i-attach-a-hand-to-an-arm
        # physics = mjcf.Physics.from_mjcf_model(end_effector.mjcf_model)
        # robot_key_frame = self._model.find("key", "home")
        # if robot_key_frame is not None:
        #     end_effector_key_frame = end_effector.mjcf_model.find("key", "home")
        #     if end_effector_key_frame is None:
        #         robot_key_frame.ctrl = np.concatenate([robot_key_frame.ctrl, np.zeros(physics.model.nu )])
        #         robot_key_frame.qpos = np.concatenate([robot_key_frame.qpos, np.zeros(physics.model.nq)])
        #     else:
        #         robot_key_frame.ctrl = np.concatenate([robot_key_frame.ctrl, end_effector_key_frame.ctrl])
        #         robot_key_frame.qpos = np.concatenate([robot_key_frame.qpos, end_effector_key_frame.qpos])

        # remove keyframe from robot or end effector
        robot_key_frame = self._model.find("key", "home")
        if robot_key_frame is not None:
            robot_key_frame.remove()

        self.attach(end_effector, self.flange)

        # update the TCP offset (bookkeeping)
        self.tcp_in_flange_pose[:3] = end_effector.tcp_offset

    @property
    def mjcf_model(self):
        return self._model

    def get_joint_positions_from_tcp_pose(self, tcp_pose: POSE_TYPE) -> Optional[JOINT_CONFIGURATION_TYPE]:
        flange_pose = self._get_flange_pose_from_tcp_pose(tcp_pose)
        joint_config = self.ik_solver.solve_ik(flange_pose, self.home_joint_positions)
        return joint_config

    def is_pose_reachable(self, tcp_pose: POSE_TYPE) -> bool:
        return self.get_joint_positions_from_tcp_pose(tcp_pose) is not None

    def _get_tcp_pose_from_flange_pose(self, flange_pose: POSE_TYPE) -> POSE_TYPE:
        flange_in_base_matrix = SE3Container.from_quaternion_and_translation(
            flange_pose[3:], flange_pose[:3]
        ).homogeneous_matrix
        tcp_in_flange_matrix = SE3Container.from_quaternion_and_translation(
            self.tcp_in_flange_pose[3:], self.tcp_in_flange_pose[:3]
        ).homogeneous_matrix
        tcp_in_base_matrix = flange_in_base_matrix @ tcp_in_flange_matrix
        tcp_in_base_se3 = SE3Container.from_homogeneous_matrix(tcp_in_base_matrix)
        tcp_in_base_pose = np.concatenate([tcp_in_base_se3.translation, tcp_in_base_se3.orientation_as_quaternion])
        return tcp_in_base_pose

    def _get_flange_pose_from_tcp_pose(self, tcp_pose: POSE_TYPE) -> POSE_TYPE:
        tcp_in_base_matrix = SE3Container.from_quaternion_and_translation(
            tcp_pose[3:], tcp_pose[:3]
        ).homogeneous_matrix
        tcp_in_flange_matrix = SE3Container.from_quaternion_and_translation(
            self.tcp_in_flange_pose[3:], self.tcp_in_flange_pose[:3]
        ).homogeneous_matrix
        flange_in_tcp_matrix = np.linalg.inv(tcp_in_flange_matrix)
        flange_in_base_matrix = tcp_in_base_matrix @ flange_in_tcp_matrix
        flange_in_base_se3 = SE3Container.from_homogeneous_matrix(flange_in_base_matrix)
        flange_in_base_pose = np.concatenate(
            [flange_in_base_se3.translation, flange_in_base_se3.orientation_as_quaternion]
        )
        return flange_in_base_pose

    def get_tcp_pose(self, physics=None) -> POSE_TYPE:
        # TODO: make sure that pose is
        # expressed in robot base frame
        # now it is in the world frame.

        flange_position = physics.named.data.site_xpos[self.flange.full_identifier]

        flange_rotation_matrix = physics.named.data.site_xmat[self.flange.full_identifier]
        flange_rotation_matrix = np.array(flange_rotation_matrix).reshape(3, 3)

        flange_se3 = SE3Container.from_rotation_matrix_and_translation(flange_rotation_matrix, flange_position)
        return np.copy(
            self._get_tcp_pose_from_flange_pose(
                np.concatenate([flange_se3.translation, flange_se3.orientation_as_quaternion])
            )
        )

    def get_joint_positions(self, physics) -> np.ndarray:
        return np.copy(physics.bind(self.joints).qpos)

    def _reset_targets(self):
        self.joint_trajectory = None

    def set_tcp_pose(self, physics: mjcf.Physics, pose: np.ndarray):
        joint_positions = self.get_joint_positions_from_tcp_pose(pose)
        if joint_positions is not None:
            self.set_joint_positions(physics, joint_positions)
        else:
            # TODO: log reset pose was not reachable.
            pass

    def set_joint_positions(self, physics: mjcf.Physics, joint_positions: np.ndarray):
        physics.bind(self.joints).qpos = joint_positions
        physics.bind(self.joints).qvel = np.zeros(self.dof)
        physics.bind(self.actuators).ctrl = joint_positions
        self._reset_targets()

    # control API

    def moveL(self, physics: mjcf.Physics, tcp_pose: np.ndarray, speed: float):
        raise NotImplementedError("moveL not implemented")

    def movej_IK(self, physics: mjcf.Physics, tcp_pose: np.ndarray, speed: float):
        if speed > self.max_joint_speed:
            print(f"required joint speed {speed} is too high for this robot.")

        target_joint_positions = self.get_joint_positions_from_tcp_pose(tcp_pose)
        if target_joint_positions is None:
            # TODO : log IK failed
            print("IK failed")
            return
        self.moveJ(physics, target_joint_positions, speed)

    def moveJ(self, physics: mjcf.Physics, target_joint_positions: np.ndarray, speed: float):
        current_joint_positions = self.get_joint_positions(physics)
        difference_vector = target_joint_positions - current_joint_positions
        time = np.max(np.abs(difference_vector)) / speed

        start_time = physics.time()
        trajectory = JointTrajectory(
            [Waypoint(current_joint_positions, start_time), Waypoint(target_joint_positions, start_time + time)]
        )
        self.joint_trajectory = trajectory

    def servoL(self, physics: mjcf.Physics, tcp_pose: np.ndarray, time: float):
        target_joint_positions = self.get_joint_positions_from_tcp_pose(tcp_pose)
        if target_joint_positions is None:
            # failsafe for IK solver not finding a solution
            print("IK failed")
            raise ValueError("IK failed")
        return self.servoJ(physics, target_joint_positions, time)

    def servoJ(self, physics, target_joint_positions, time: float):
        """This implementation differs from the UR implementaion.

        They use an additional PD controller to determine the setpoint for the joint positions at each timestep.
        This implementation just lineary interpolates the joint positions between the current and target positions
        at the desired speed and uses those as setpoints.

        The error dynamics of the UR implementation are better, but this implementation is simpler as it
        does not require tuning the additional PD controller and has the robot move at constant velocity.

        Args:
            physics (_type_): _description_
            target_joint_positions (_type_): _description_
            time (float): _description_
        """

        self._reset_targets()

        time = time
        current_joint_positions = self.get_joint_positions(physics)

        difference_vector = target_joint_positions - current_joint_positions
        # speed is defined as largest joint movement divided by time
        speed = np.max(np.abs(difference_vector)) / time

        if speed > self.max_joint_speed:
            print(f"required joint speed {speed} is too high for this robot.")

        start_time = physics.time()
        trajectory = JointTrajectory(
            [Waypoint(current_joint_positions, start_time), Waypoint(target_joint_positions, start_time + time)]
        )
        self.joint_trajectory = trajectory

    def before_substep(self, physics: mjcf.Physics, random_state):
        if self.joint_trajectory is not None:
            physics.bind(self.actuators).ctrl = self.joint_trajectory.get_target_joint_positions(physics.time())

            # log low-level joint data
            # TODO: should create wrapper around robot to do logging instead of doing it here.
            # bc it slows down the simulation..
            # self.joint_position_history.append(self.get_joint_positions(physics))
            # self.joint_target_history.append(self.joint_trajectory.get_target_joint_positions(physics.time()))

            if self.joint_trajectory.is_finished(physics.timestep()):
                self.joint_trajectory = None

    def is_moving(self) -> bool:
        return self.joint_trajectory is not None

    # def _build_observables(self):
    #     # joint positions, joint velocities
    #     # tcp position, tcp velocities
    #     # tcp F/T
    #     # how to deal with SE2 vs SE3?

    def _build_observables(self):
        return RobotObservables(self)


class RobotObservables(composer.Observables):
    @composer.observable
    def tcp_pose(self):
        return observable.Generic(self._entity.get_tcp_pose)

    @composer.observable
    def tcp_position(self):
        return observable.Generic(lambda physics: self._entity.get_tcp_pose(physics)[:3])


class UR5e(Robot):

    # obtained manually from the XML
    _BASE_BODY_NAME = "base"
    _FLANGE_SITE_NAME = "attachment_site"

    home_joint_positions = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, -0.5]) * np.pi
    max_joint_speed = 1.0  # rad/s - https://store.clearpathrobotics.com/products/ur3e

    def __init__(self):
        from robot_descriptions import ur5e_mj_description

        self._MODEL_XML_PATH = ur5e_mj_description.MJCF_PATH
        solver = AnalyticURIKsolver("ur5e")
        super().__init__(solver)

    def _build(self):
        ret = super()._build()
        # get the robot model and change the base orientation
        self._model.find("body", "base").quat = [0, 0, 0, -1]
        return ret


def test_ur5e_position_controllers_servoL():
    import matplotlib.pyplot as plt

    frequency = 20

    robot = UR5e()
    model = robot.mjcf_model
    physics = mjcf.Physics.from_mjcf_model(model)

    with physics.reset_context():
        robot.set_joint_positions(physics, robot.home_joint_positions)

    for i in range(5):
        speed = 0.1
        delta_step = np.array(
            [
                -speed / frequency * (1) ** i,
                speed / frequency,
                0.1 / frequency * (-1) ** i,
                0,
                speed / frequency,
                (-1) ** i * speed / frequency,
                0,
            ]
        )
        delta_step = np.array([speed / frequency, 0, 0, 0, 0, 0, 0])
        # delta_step = np.zeros(7) # this tests
        robot.servoL(physics, robot.get_tcp_pose(physics) + delta_step, 1 / frequency)
        robot.before_step(physics, None)
        for i in range(int(500 * 1 / frequency)):
            robot.before_substep(physics, None)
            physics.step()

    # create a plot with 6 figures: on each figure the target and actual joints is plotted
    fig, axs = plt.subplots(6, 1, sharex=True)
    # create title for the whole figure
    fig.suptitle(f"Low-level joint position control under servoL @ {frequency} Hz")
    print(robot.joint_position_history)
    for i in range(6):
        axs[i].plot(np.array(robot.joint_target_history)[:, i])
        axs[i].plot(np.array(robot.joint_position_history)[:, i])
        axs[i].set_ylabel(f"{robot.joints[i].name}")
        # add legend to the plot to show which line is the target and the actual position
        axs[i].legend(["target", "actual"])
        axs[i].set_xlabel("physics steps")

    plt.show()


if __name__ == "__main__":

    robot = UR5e()
    physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)

    # set physics step size
    print(f"physics timestep: {physics.timestep()}")

    # force robot to move to a specific pose
    joints = np.array([0.0, -0.5, 0.5, -0.5, -0.5, -0.5]) * np.pi
    robot.set_joint_positions(physics, joints)

    target_pose = np.array([0.1, 0.3, 0.5, 1, 0, 0, 0])

    print(f"initial pose: {robot.get_tcp_pose(physics)}")
    for _ in range(20):
        robot.servoL(physics, target_pose, 0.2)
        for _ in range(100):
            robot.before_substep(physics, None)
            physics.step()

    print(f"final pose: {robot.get_tcp_pose(physics)}")
    print(f"target pose: {target_pose}")
