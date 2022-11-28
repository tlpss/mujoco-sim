from __future__ import annotations

import numpy as np
from dm_control import composer, mjcf


class UR5e(composer.Entity):
    """
    Robot
    """

    XML_PATH = ""  # TODO: submodule menagerie & link the XML
    _BASE_ELEMENT = None
    _TCP_ELEMENT = None

    def __init__(self) -> None:
        """
        _model: the xml tree is built top-down, to make sure that you always have acces to entire scene so far,
        including the worldbody (for creating mocaps, which have to be children of the worldbody)
        or can define equalities with other elements.
        """

        self.physics = None
        # instantiate a MJCF model to start building on.
        self._model = mjcf.RootElement(self._ROOT_ELEMENT_NAME)
        # call the super init, which handles the building
        super().__init__()

    def _build(self):
        pass

    def initialize_episode(self, physics, random_state):
        self.physics = physics
        return super().initialize_episode(physics, random_state)

    @property
    def mjcf_model(self):
        return self._model

    def get_tcp_pose(self, physics=None) -> np.ndarray:
        pass

    def get_joint_configuration(self, physics) -> np.ndarray:
        pass

    def set_tcp_pose(self, physics: mjcf.Physics, position: np.ndarray):
        pass

    # control API
    def set_tcp_target_pose(self, physics: mjcf.Physics, target_position: np.ndarray):
        pass

    def before_substep(self, physics, random_state):
        # TODO: convert the TCP target position to the desired joint target positions
        # and then apply these target positions to the robot (interpolate until the next tcp command!)
        pass

    def _build_observables(self):
        # joint positions, joint velocities
        # tcp position, tcp velocities
        # tcp F/T
        pass


if __name__ == "__main__":

    robot = UR5e()
    model = robot.mjcf_model

    physics = mjcf.Physics.from_mjcf_model(model)
    robot.initialize_episode(physics, None)
    for _ in range(50):  # default timestep  = 2ms
        # set the mocap target position
        robot.set_target_position(physics, np.array([0.05, 0.04]))
        print(f"{physics.named.data.xpos}")
        physics.step()

    # check if both are equal
    print(robot.get_position())
    print(robot.observables.position(physics))
    # check if reset works

    with physics.reset_context():
        robot.reset_position(physics, np.array([0.12, 0.13]))
    print(robot.observables.position(physics))
