from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from airo_core.spatial_algebra.se3 import SE3Container
from dm_control import composer, mjcf

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
        pose[3:] += np.random.randn(4) * 0.01
        targj = solver.inverse(pose, q_guess=q_guess)
        if targj is not None:
            break

    if targj is None:
        warnings.warn("IKFast failed... most likely the pose is out of reach of the robot?")
    return targj


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

    def __init__(self) -> None:
        """
        _model: the xml tree is built top-down, to make sure that you always have acces to entire scene so far,
        including the worldbody (for creating mocaps, which have to be children of the worldbody)
        or can define equalities with other elements.
        """

        self.physics = None
        self._model = None
        self.joint_speed = 1  # rad/s
        self.tcp_in_flange_pose: POSE_TYPE = np.array(
            [0, 0, 0, 0, 0, 0, 1.0]
        )  # Tool Center Frame translation wrt to the FLANGE (IK etc is based on this frame)

        self.tcp_target_pose = None
        self.joint_target_positions = None

        self.ik_fast_solver = ur_kinematics.URKinematics("ur5e")

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
        return self._get_tcp_pose_from_flange_pose(
            np.concatenate([flange_se3.translation, flange_se3.get_orientation_as_quaternion()])
        )

    def get_joint_positions(self, physics) -> np.ndarray:
        return physics.bind(self.joints).qpos

    def _reset_targets(self):
        self.tcp_target_pose = None
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
        physics.bind(self.actuators).ctrl = joint_positions

        self._reset_targets()

    # control API
    def set_tcp_target_pose(self, physics: mjcf.Physics, tcp_pose: np.ndarray):
        if self.is_pose_reachable(tcp_pose):
            self._reset_targets()
            self.tcp_target_pose = tcp_pose
        else:
            # TODO: log!
            print("TCP pose is not reachable!")

    def before_step(self, physics, random_state):

        # initialize the setpoints for interpolation with the current joint positions / tcp pose
        self.joint_setpoint = self.get_joint_positions(physics)
        self.tcp_setpoint = self.get_tcp_pose(physics)

        return super().before_step(physics, random_state)

    def before_substep(self, physics: mjcf.Physics, random_state):

        if self.tcp_target_pose is not None:
            # implement tcp movements as 'movej_IK' for now
            # but should do moveL in the future.
            self.joint_target_positions = self.get_joint_positions_from_tcp_pose(self.tcp_target_pose)
            self.tcp_target_pose = None

        if self.joint_target_positions is not None:

            # note that this is w.r.t. the current setpoint.
            difference_vector = self.joint_target_positions - self.joint_setpoint
            if np.linalg.norm(difference_vector, 1) < 1e-2:
                self.joint_target_positions = None
                return

            difference_vector /= np.linalg.norm(difference_vector, 1)
            difference_vector *= physics.timestep() * self.joint_speed
            self.joint_setpoint += difference_vector
            physics.bind(self.actuators).ctrl = self.joint_setpoint

    def is_moving(self) -> bool:
        return self.tcp_target_pose is not None or self.joint_target_positions is not None

    # def _build_observables(self):
    #     # joint positions, joint velocities
    #     # tcp position, tcp velocities
    #     # tcp F/T
    #     # how to deal with SE2 vs SE3?


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    robot = UR5e()
    print(f" flange home pos according to ikfast: {robot.ik_fast_solver.forward(np.zeros(6))}")
    model = robot.mjcf_model
    physics = mjcf.Physics.from_mjcf_model(model)

    with physics.reset_context():
        robot.set_joint_positions(physics, robot.home_joint_positions)
    print(robot.get_tcp_pose(physics))
    plt.imshow(physics.render())
    plt.show()

    # robot.set_tcp_target_pose(physics, np.array([0.4, -0.3, 0.2, 0, 1, 0, 0]))
    for i in range(500 * 20):
        robot.before_substep(physics, None)
        physics.step()
        if i % 500 == 0:
            print(robot.get_tcp_pose(physics))
            print(f"joint pos = {robot.get_joint_positions(physics)}")
            plt.imshow(physics.render())
            plt.show()
