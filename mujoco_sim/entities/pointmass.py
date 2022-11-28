from __future__ import annotations

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable

from mujoco_sim.entities.utils import build_mocap


class PointMass2D(composer.Entity):
    """
    A very simple Pointmass Entity:

    the MJCF model is a single-body sphere that is connected with a free floating joint to the worldbody.
    Free floating objects can be controlled directly by setting their position (but this 'skips' the physics engine during the movement),
    or by welding them to  mocap bodies, or by creating arbitrary joints (mjc allows for multiple joints between two elements),
        e.g see dm control suite https://github.com/deepmind/dm_control/blob/main/dm_control/suite/point_mass.xml.
        also note that they use 2 tendons to control x-y (tendon -> actuator)


    Entities must define the `_build()` method and the `mjcf_model` property.

    Additionally they can define observables and implement some of the many callbacks from section 5.3 of the dm_control paper
    """

    _ROOT_ELEMENT_NAME = "pointmass"

    def __init__(self, radius: float = 0.02, mass: float = 0.1, mocap=None) -> None:
        """
        _model: the xml tree is built top-down, to make sure that you always have acces to entire scene so far,
        including the worldbody (for creating mocaps, which have to be children of the worldbody)
        or can define equalities with other elements.
        """
        self.mass = mass
        self.radius = radius
        self.mocap = mocap

        self.physics = None
        # instantiate a MJCF model to start building on.
        self._model = mjcf.RootElement(self._ROOT_ELEMENT_NAME)
        # call the super init, which handles the building
        super().__init__()

    def _build(self):
        """
        This function has the responsability to build the MJCF description of this entity.

        It can do this procedurally, as is done here, or it can do this by loading a static XML file.
        """

        # the actual pointmass
        self.pointmass: mjcf.Element = self._model.worldbody.add(
            "body", name="pointmass_body", pos=[0.0, 0.0, self.radius]
        )
        self.pointmass.add("geom", type="sphere", size=[self.radius], mass=self.mass, rgba=[255, 0, 0, 0.5])

        # free joint would be the easiest option, but these required the body to be child of worldbody, which is incompatible
        # with creating a tree of independent Entities.
        # self.freejoint = self.pointmass.add("freejoint",name="pointmass-freejoint")
        # you would have to do arena.attach(entity).add('freejoint') each time you add the entity in a tree
        # instead of the canonical arena.attach(entity) and you would not be able to just instantiate the entity for testing
        # this is what they do in the dm_control paper for the creatures.

        # so create slide joints for X,Y positions (in 6DOF you have to add balljoint and Z slider)
        self.x_joint: mjcf.Element = self.pointmass.add("joint", name="pointmass_x", type="slide", axis=[1, 0, 0])
        self.y_joint: mjcf.Element = self.pointmass.add("joint", name="pointmass_y", type="slide", axis=[0, 1, 0])

        if self.mocap is None:
            self.mocap = build_mocap(self._model, "pointmass_mocap")
            # weld body to mocap to have it track the mocap
            # this way the teleporting has no impact on the physics.
            self._model.equality.add(
                "weld",
                name="mocap_to_mass_weld",
                body1=self.mocap.full_identifier,
                body2=self.pointmass.full_identifier,
            )

    def initialize_episode(self, physics, random_state):
        self.physics = physics
        return super().initialize_episode(physics, random_state)

    @property
    def mjcf_model(self):
        return self._model

    def get_position(self, physics=None) -> np.ndarray:
        """get the current position of the pointmass

        Args:
            physics (mjcf.Physics): _description_

        Returns:
            (np.ndarray): [X,Y]
        """
        physics = physics or self.physics
        assert physics is not None
        return physics.bind(self.pointmass).xpos[:2]

    def reset_position(self, physics: mjcf.Physics, position: np.ndarray):
        """Sets the position of the pointmass directy ('teleporting')
        by modifying the qpos values

        Args:
            position (np.ndarray): 2D x,y position
            physics (mjcf.Physics): _description_
        """

        assert position.shape == (2,)
        # setting the state directly can only be done with qpos,qvel; but that would
        # influence the physics (you cannot directly set cartesian positions as in e.g. pybullet,
        # bc mujoco operates in reduced coordinates)
        # actuation input is given through 'mjData.ctrl', or via `data.mocap`.

        # so you always have to do the 'inverse kinematics' from cartesian space to joint space
        # but since we have two slide joints, the inverse kinematics are simply q1,q2 = x,y

        # use full identifier to make sure that after attaching to other elements
        # the joint name still reflects the xml joint name
        # arena.attach(model) will add a namespace for alle mjcf elements in the model
        # x_joint.name will still be 'pointmass_x', although the element in the xml
        # will be named '<model_name>/<joint_name>' so using .name would result in an error.
        physics.named.data.qpos[self.x_joint.full_identifier] = position[0]
        physics.named.data.qpos[self.y_joint.full_identifier] = position[1]

        physics.named.data.qvel[self.x_joint.full_identifier] = 0.0
        physics.named.data.qvel[self.y_joint.full_identifier] = 0.0

        # also reset the mocap
        self.set_target_position(physics, position)

    # control API
    def set_target_position(self, physics: mjcf.Physics, target_position: np.ndarray):
        """_summary_

        Args:
            physics (mjcf.Physics): _description_
            target_position (np.ndarray): _description_
        """
        # the bind function is for convenient access to the mujoco mData class that contains all
        # dynamic variables of the simulation

        # pysics.bind(pointmass.mocap).pos = target_position
        # DO NOT USE THAT here: it returns a view on the derrived physics state, not on mocap_pos
        # and you cannot use that derrived view to set the mocap input.

        assert target_position.shape == (2,)
        physics.named.data.mocap_pos[self.mocap.name][:2] = target_position

    def _build_observables(self):
        """build the observables. This a the dm_control way to structure the code for generating observations
        each observable can then be turned ON or OFF in a `task`. random delays, aggregations, distorters etc can also be applied
        later on.
        """
        return PointMassObservables(self)


class PointMassObservables(composer.Observables):
    """Observables for pointmas 2D"""

    _entity: PointMass2D  # typing

    # if you want to expose joints, this is how to do it:

    # @composer.observable
    # def joint_positions(self) -> observable.MJCFFeature:
    #     return observable.MJCFFeature(kind="qpos",mjcf_element=[self._entity.x_joint,self._entity.y_joint])

    @composer.observable
    def position(self) -> observable.Generic:
        obs = observable.Generic(raw_observation_callable=self._entity.get_position)
        # obs.array_spec = specs.Array((2,),np.float32)
        return obs


if __name__ == "__main__":

    pointmass = PointMass2D()
    model = pointmass.mjcf_model

    physics = mjcf.Physics.from_mjcf_model(model)
    pointmass.initialize_episode(physics, None)
    for _ in range(50):  # default timestep  = 2ms
        # set the mocap target position
        pointmass.set_target_position(physics, np.array([0.05, 0.04]))
        print(f"{physics.named.data.xpos}")
        physics.step()

    # check if both are equal
    print(pointmass.get_position())
    print(pointmass.observables.position(physics))
    # check if reset works

    with physics.reset_context():
        pointmass.reset_position(physics, np.array([0.12, 0.13]))
    print(pointmass.observables.position(physics))
