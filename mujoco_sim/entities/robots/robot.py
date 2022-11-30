from __future__ import annotations

from typing import Optional

import numpy as np
from dm_control import composer, mjcf

from mujoco_sim.entities.utils import get_assets_root_folder
from ur_ikfast.ur_ikfast import ur_kinematics

ur5_ikfast_solver = ur_kinematics.URKinematics("ur5e")
from mujoco_sim.type_aliases import JOINT_CONFIGURATION_TYPE, POSE_TYPE


def solve_ik_ikfast(pose: POSE_TYPE, q_guess: JOINT_CONFIGURATION_TYPE) -> Optional[JOINT_CONFIGURATION_TYPE]:
    """analytical (global) IK, returns the solution that is closest to the current joint configuration
    without taking collisions into account.
    """
    targj = None
    for _ in range(6):
        # fix for axis-aligned orientations (which is often used in e.g. top-down EEF orientations
        # add random noise to the EEF orienation to avoid axis-alignment
        # see https://github.com/cambel/ur_ikfast/issues/4
        pose[3:] += np.random.randn(4) * 0.01
        targj = ur5_ikfast_solver.inverse(pose, q_guess=q_guess)
        if targj is not None:
            break

    if targj is None:
        raise Warning("IKFast failed... most likely the pose is out of reach of the robot?")
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

    def __init__(self) -> None:
        """
        _model: the xml tree is built top-down, to make sure that you always have acces to entire scene so far,
        including the worldbody (for creating mocaps, which have to be children of the worldbody)
        or can define equalities with other elements.
        """

        self.physics = None
        self._model = None
        self.tcp_pose: POSE_TYPE = np.array(
            [0, 0, 0, 0, 0, 0, 1.0]
        )  # Tool Center Frame translation wrt to the FLANGE (IK etc is based on this frame)

        super().__init__()

    def _build(self):
        self._model = mjcf.from_path(str(get_assets_root_folder() / self.XML_PATH))
        self.joints = self._model.find_all("joint")
        self.dof = len(self.joints)
        self.base_element = self._model.find("body", self._BASE_BODY_NAME)
        self.flange = self._model.find("site", self._FLANGE_SITE_NAME)

    def initialize_episode(self, physics, random_state):
        self.physics = physics
        return super().initialize_episode(physics, random_state)

    @property
    def mjcf_model(self):
        return self._model

    def get_tcp_pose(self, physics=None) -> POSE_TYPE:
        flange_position = physics.named.data.site_xpos[self._FLANGE_SITE_NAME]
        flange_orientation = physics.named.data.site_xquat[self._FLANGE_SITE_NAME]

        # TODO: convert quaternion from scalar first to scalar last.
        # TODO: get tcp_in_base
        return flange_orientation, flange_position

    def get_joint_positions(self, physics=None) -> np.ndarray:
        return physics.data.qpos

    def set_tcp_pose(self, physics: mjcf.Physics, pose: np.ndarray):
        # requires global IK
        # transform TCP pose to flange pose
        # do ik on flange pose
        # set joint positions
        pass

    def set_joint_positions(self, physics: mjcf.Physics, joint_positions: np.ndarray):
        physics.data.qpos = joint_positions
        physics.data.qvel = np.zeros(self.dof)

    # control API
    def set_tcp_target_pose(self, physics: mjcf.Physics, target_position: np.ndarray):
        # transform tcp pose to flange pose
        # do IK
        # set the joint targets for the controller.
        pass

    def before_substep(self, physics, random_state):
        # TODO: convert the TCP target position to the desired joint target positions
        # and then apply these target positions to the robot

        # maybe even interpolate until the next tcp command? this requires knowledge on the control rate.
        pass

    # def _build_observables(self):
    #     # joint positions, joint velocities
    #     # tcp position, tcp velocities
    #     # tcp F/T
    #     pass


if __name__ == "__main__":

    robot = UR5e()
    model = robot.mjcf_model

    physics = mjcf.Physics.from_mjcf_model(model)
    print(physics.named.data.site_xpos)
    print(physics.named.data.xpos)
    print(robot.flange)
    print(robot.get_tcp_pose(physics))
