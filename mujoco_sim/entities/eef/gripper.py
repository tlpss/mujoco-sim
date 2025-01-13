import abc

import numpy as np
from dm_control import mjcf
from robot_descriptions import robotiq_2f85_mj_description

from mujoco_sim.entities.eef.cylinder import EEF
from mujoco_sim.entities.utils import write_xml
from mujoco_sim.type_aliases import VECTOR_TYPE


class ParallelGripper(EEF):
    @property
    @abc.abstractmethod
    def open_distance(self):
        """distance between the fingers when the gripper is open (in meters)"""
        raise NotImplementedError

    def __init__(self) -> None:
        super().__init__()
        self._joint_target = 0.0

    # TODO: extract all relevant functions from the gripper class once it is stable.

    def close(self, physics):
        self.move(physics, 0.0)

    def open(self, physics):
        self.move(physics, self.open_distance)

    @property
    def actuator(self):
        raise NotImplementedError


class Robotiq2f85(ParallelGripper):
    xml_path = robotiq_2f85_mj_description.MJCF_PATH
    max_driver_joint_angle = 0.8

    joint_home_positions = np.array([0.0, 0.0, 0.005, -0.01, 0.0, 0.0, 0.005, -0.01])

    def _build(self, *args, **kwargs):
        self._model = mjcf.from_path(self.xml_path)
        self._model.worldbody.add("site", name="tcp-site", pos=[0.0, 0.0, 0.174], size=[0.001])

    @property
    def tcp_offset(self) -> VECTOR_TYPE:
        return np.array([0.0, 0.0, 0.174])

    @property
    def open_distance(self):
        return 0.085

    @property
    def mjcf_model(self):
        return self._model

    @property
    def actuator(self):
        return self._model.find("actuator", "fingers_actuator")

    @property
    def joints(self):
        # these two are driven, others are held by equality constraints
        return [self._model.find("joint", "left_driver_joint"), self._model.find("joint", "right_driver_joint")]

    def _joint_angle_to_finger_distance(self, joint_angle):
        # hacky. joint range of the drivers = 0.8 radians
        # so approximate the linear distance between the fingers (sin(x) = x)
        # by the joint angle scaled to the linear range of the gripper
        return self.open_distance * (1 - np.sin(joint_angle) / np.sin(self.max_driver_joint_angle))

    def _finger_distance_to_joint_angle(self, finger_distance):
        return np.arcsin((1 - finger_distance / self.open_distance) * np.sin(self.max_driver_joint_angle))

    def get_finger_opening(self, physics):
        return self._joint_angle_to_finger_distance(physics.bind(self.joints[0]).qpos)

    def move(self, physics, finger_distance: float):
        joint_angle = self._finger_distance_to_joint_angle(finger_distance)
        actuator_control_value = (
            joint_angle / self.max_driver_joint_angle * 255
        )  # 255 is the max value for the actuator
        physics.bind(self.actuator).ctrl = actuator_control_value

    def reset(self, physics):
        """resets the gripper and all joints to the 'open' position

        Args:
            physics (_type_): _description_
        """
        physics.bind(self._model.find_all("joint")).qpos = self.joint_home_positions
        physics.bind(self.actuator).ctrl = 0.0

    def is_moving():
        # required for synchronous mode?
        # can check qvel for a number of steps if we keep track of them?
        raise NotImplementedError


if __name__ == "__main__":
    gripper = Robotiq2f85()
    physics = mjcf.Physics.from_mjcf_model(gripper.mjcf_model)

    with physics.reset_context():
        gripper.reset(physics)

    gripper.move(physics, 0.01)
    for _ in range(2000):
        physics.step()
    print(physics.named.data.xpos)
    print(gripper.get_finger_opening(physics))

    write_xml(gripper.mjcf_model)
