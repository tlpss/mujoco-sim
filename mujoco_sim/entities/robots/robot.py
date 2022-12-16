from __future__ import annotations

import dataclasses
import warnings
from collections import deque
from typing import Optional

import numpy as np
from airo_core.spatial_algebra.se3 import SE3Container
from dm_control import composer, mjcf
from dm_control.composer.observation import observable

from mujoco_sim.entities.eef.cylinder import EEF
from mujoco_sim.entities.utils import get_assets_root_folder
from mujoco_sim.type_aliases import JOINT_CONFIGURATION_TYPE, POSE_TYPE
from ur_ikfast.ur_ikfast import ur_kinematics


def solve_ik_ikfast(
    solver: ur_kinematics.URKinematics, pose: POSE_TYPE, q_guess: JOINT_CONFIGURATION_TYPE
) -> Optional[JOINT_CONFIGURATION_TYPE]:
    """analytical (global) IK, returns the solution that is closest to the current joint configuration
    without taking collisions into account.
    """
    targj = None
    for _ in range(6):
        # fix for axis-aligned orientations (which is often used in e.g. top-down EEF orientations
        # add random noise to the EEF orienation to avoid axis-alignment
        # see https://github.com/cambel/ur_ikfast/issues/4
        pose[3:] += np.random.randn(4) * 0.001
        targj = solver.inverse(pose, q_guess=q_guess)
        if targj is not None:
            break

    if targj is None:
        warnings.warn("IKFast failed... most likely the pose is out of reach of the robot?")
    return targj


@dataclasses.dataclass
class Waypoint:
    joint_positions: JOINT_CONFIGURATION_TYPE
    timestep: float


class JointTrajectory:
    """A container to hold a joint trajectory defined by a number of (key) waypoints.
    The trajectory is defined by linear interpolation between the waypoints.

    Note that this might not be a smooth trajectory, as the joint velocities/accelerations are not guarantueed to be continuous
    for the interpolated trajectory. To overcome this, use an appropriate trajectory generator and timestep to provide the
    waypoints in this trajectory so that the discontinuities that are introduced by the linear interpolation are small.
    """

    def __init__(self, waypoints: list[Waypoint]):
        # # sort by timestep to make sure that the waypoints are in order
        self.waypoints = sorted(waypoints, key=lambda x: x.timestep)

    def get_nearest_waypoints(self, t: float) -> tuple[Waypoint, Waypoint]:
        for i in range(len(self.waypoints) - 1):
            if t >= self.waypoints[i].timestep and t <= self.waypoints[i + 1].timestep:
                return self.waypoints[i], self.waypoints[i + 1]
        raise ValueError("should not be here")

    def _clip_timestep(self, t: float) -> float:
        """clips the timestep to the range of the waypoints"""
        return np.clip(t, self.waypoints[0].timestep, self.waypoints[-1].timestep)

    def get_target_joint_positions(self, t: float) -> JOINT_CONFIGURATION_TYPE:
        """returns the target joint positions at time t by linear interpolation between the waypoints"""
        t = self._clip_timestep(t)
        previous_waypoint, next_waypoint = self.get_nearest_waypoints(t)
        t0, q0 = previous_waypoint.timestep, previous_waypoint.joint_positions
        t1, q1 = next_waypoint.timestep, next_waypoint.joint_positions
        return q0 + (q1 - q0) * (t - t0) / (t1 - t0)

    def get_target_joint_velocities(self, t: float) -> JOINT_CONFIGURATION_TYPE:
        """returns the target joint velocities at time t by linear interpolation between the waypoints"""
        t = self._clip_timestep(t)
        previous_waypoint, next_waypoint = self.get_nearest_waypoints(t)
        t0, q0 = previous_waypoint.timestep, previous_waypoint.joint_positions
        t1, q1 = next_waypoint.timestep, next_waypoint.joint_positions
        return (q1 - q0) / (t1 - t0)

    def is_finished(self, t: float) -> bool:
        """returns True if the trajectory is finished at time t"""
        return t >= self.waypoints[-1].timestep


class UR5e(composer.Entity):
    """
    Robot
    """

    XML_PATH = "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"

    # defines the base frame
    _BASE_BODY_NAME = "base"

    # site that defines the frame to attach EEF
    #  (and is used internally for IK as it is the last element in the robot kinematics tree)
    _FLANGE_SITE_NAME = "attachment_site"

    home_joint_positions = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, -0.5]) * np.pi
    max_joint_speed = 1.0  # rad/s - https://store.clearpathrobotics.com/products/ur3e

    def __init__(self) -> None:
        """
        _model: the xml tree is built top-down, to make sure that you always have acces to entire scene so far,
        including the worldbody (for creating mocaps, which have to be children of the worldbody)
        or can define equalities with other elements.
        """

        self.physics = None
        self._model = None
        self.tcp_in_flange_pose: POSE_TYPE = np.array(
            [0, 0, 0, 0, 0, 0, 1.0]
        )  # Tool Center Frame translation wrt to the FLANGE (IK etc is based on this frame)

        self.joint_trajectory: Optional[JointTrajectory] = None

        self.commanded_joint_positions = np.zeros(6)

        self.ik_fast_solver = ur_kinematics.URKinematics("ur5e")

        # used for visualization of the performances of the low-level controllers
        self.joint_position_history = deque(maxlen=10000)
        self.joint_target_history = deque(maxlen=10000)

        self._reset_targets()
        super().__init__()

    def _build(self):
        self._model = mjcf.from_path(str(get_assets_root_folder() / self.XML_PATH))
        self.joints = self._model.find_all("joint")
        self.actuators = self._model.find_all("actuator")
        self.dof = len(self.joints)
        self.base_element = self._model.find("body", self._BASE_BODY_NAME)
        self.flange: mjcf.Element = self._model.find("site", self._FLANGE_SITE_NAME)

    def initialize_episode(self, physics, random_state):
        self.physics = physics
        return super().initialize_episode(physics, random_state)

    def attach_end_effector(self, end_effector: EEF):
        self.attach(end_effector, self.flange)
        self.tcp_in_flange_pose[:3] = end_effector.tcp_offset

    @property
    def mjcf_model(self):
        return self._model

    def get_joint_positions_from_tcp_pose(self, tcp_pose: POSE_TYPE) -> Optional[JOINT_CONFIGURATION_TYPE]:
        flange_pose = self._get_flange_pose_from_tcp_pose(tcp_pose)
        joint_config = solve_ik_ikfast(self.ik_fast_solver, flange_pose, self.home_joint_positions)
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
        tcp_in_base_pose = np.concatenate(
            [tcp_in_base_se3.translation, tcp_in_base_se3.get_orientation_as_quaternion()]
        )
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
            [flange_in_base_se3.translation, flange_in_base_se3.get_orientation_as_quaternion()]
        )
        return flange_in_base_pose

    def get_tcp_pose(self, physics=None) -> POSE_TYPE:
        # TODO: make sure that pose is expressed in robot base frame
        # now it is in the world frame.

        flange_position = physics.named.data.site_xpos[self.flange.full_identifier]

        flange_rotation_matrix = physics.named.data.site_xmat[self.flange.full_identifier]
        flange_rotation_matrix = np.array(flange_rotation_matrix).reshape(3, 3)
        flange_se3 = SE3Container.from_rotation_matrix_and_translation(flange_rotation_matrix, flange_position)
        return np.copy(
            self._get_tcp_pose_from_flange_pose(
                np.concatenate([flange_se3.translation, flange_se3.get_orientation_as_quaternion()])
            )
        )

    def get_joint_positions(self, physics) -> np.ndarray:
        return np.copy(physics.bind(self.joints).qpos)

    def _reset_targets(self):
        self.joint_trajectory = None
        self.joint_target_positions = None

    def set_tcp_pose(self, physics: mjcf.Physics, pose: np.ndarray):
        flange_pose = self._get_flange_pose_from_tcp_pose(pose)
        joint_positions = solve_ik_ikfast(self.ik_fast_solver, flange_pose, self.home_joint_positions)
        if joint_positions is not None:
            self.set_joint_positions(physics, joint_positions)
        else:
            # TODO: log reset pose was not reachable.
            pass

    def set_joint_positions(self, physics: mjcf.Physics, joint_positions: np.ndarray):
        physics.bind(self.joints).qpos = joint_positions
        physics.bind(self.joints).qvel = np.zeros(self.dof)
        physics.data.ctrl = joint_positions
        self._reset_targets()

    # control API

    def moveL(self, physics: mjcf.Physics, tcp_pose: np.ndarray, speed: float):
        raise NotImplementedError

    def movej_IK(self, physics: mjcf.Physics, tcp_pose: np.ndarray, speed: float):
        target_joint_positions = self.get_joint_positions_from_tcp_pose(tcp_pose)
        if target_joint_positions is not None:
            self.joint_target_positions = target_joint_positions

    def servoL(self, physics: mjcf.Physics, tcp_pose: np.ndarray, time: float):
        target_joint_positions = self.get_joint_positions_from_tcp_pose(tcp_pose)
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

        assert speed < self.max_joint_speed, f"required joint speed {speed} is too high for this robot."

        start_time = physics.time()
        trajectory = JointTrajectory(
            [Waypoint(current_joint_positions, start_time), Waypoint(target_joint_positions, start_time + time)]
        )
        self.joint_trajectory = trajectory

    def before_substep(self, physics: mjcf.Physics, random_state):
        if self.joint_trajectory is not None:
            physics.bind(self.actuators).ctrl = self.joint_trajectory.get_target_joint_positions(physics.time())

            # log low-level joint data
            self.joint_position_history.append(self.get_joint_positions(physics))
            self.joint_target_history.append(self.joint_trajectory.get_target_joint_positions(physics.time()))

            if self.joint_trajectory.is_finished(physics.timestep()):
                self.joint_trajectory = None

    def is_moving(self) -> bool:
        return self.tcp_target_pose is not None or self.joint_target_positions is not None

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


def test_ur5e_position_controllers_servoL():
    frequency = 10

    robot = UR5e()
    print(f" flange home pos according to ikfast: {robot.ik_fast_solver.forward(np.zeros(6))}")
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
    for i in range(6):
        axs[i].plot(np.array(robot.joint_target_history)[:, i])
        axs[i].plot(np.array(robot.joint_position_history)[:, i])
        axs[i].set_ylabel(f"{robot.joints[i].name}")
        # add legend to the plot to show which line is the target and the actual position
        axs[i].legend(["target", "actual"])
        axs[i].set_xlabel("physics steps")

    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_ur5e_position_controllers_servoL()
