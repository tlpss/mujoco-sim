import numpy as np
from dm_control import composer, mjcf


class PointMassObservables(composer.Observables):
    pass


class PointMass(composer.Entity):
    """
    A very simple Pointmass Entity:

    this is a single-body sphere that is connected with a free floating joint to the worldbody.
    Free floating objects can be controlled directly by setting their position, but this 'skips' the physics engine during the movement
    or by welding them to  mocap body.

    The mocap body itself is not considered during collisions


    Entities must define the `_build()` method and the `mjcf_model` property.
    Additionally they can define observables and implement some of the many callbacks from section 5.3 of the dm_control paper
    """

    def __init__(self, radius: float = 0.02, mass: float = 0.1, parent: mjcf.Element = None) -> None:
        """
        parent: the xml tree is built top-down, to make sure that you always have acces to entire scene so far,
        including the worldbody (for creating mocaps, which have to be children of the worldbody)
        or can define equalities with other elements.
        """
        self.mass = mass
        self.radius = radius
        self.world = parent if parent is not None else mjcf.RootElement()

        # call the super init, which handles the building
        super().__init__()

    def _build(self):
        """
        This function has the responsability to build the MJCF description of this entity.

        It can do this procedurally, as is done here, or it can do this by loading a static XML file.

        note that a 2D pointmass could be created better with 2 joints, cf dm control suite https://github.com/deepmind/dm_control/blob/main/dm_control/suite/point_mass.xml
        also note that they use 2 tendons to control x-y (tendon -> actuator)
        """

        # mocap info
        # https://github.com/deepmind/mujoco/issues/433
        # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=mocap#mocap-bodies
        self.mocap = self.world.worldbody.add("body", name="pointmass_mocap", pos=[0.0, 0.0, self.radius], mocap=True)
        self.mocap.add("site", type="sphere", size=[0.001])
        self.pointmass = self.world.worldbody.add("body", name="pointmass", pos=[0.0, 0.0, self.radius])
        self.pointmass.add("geom", type="sphere", size=[self.radius], mass=self.mass, rgba=[255, 0, 0, 0.5])
        self.pointmass.add("freejoint")
        # weld body to mocap
        self.world.equality.add("weld", name="mocap_to_mass_weld", body1=self.mocap.name, body2=self.pointmass.name)
        # avoid collisions between the
        # self.world.contact.add("exculde",self.mocap.name,self.pointmass.name)

    @property
    def mjcf_model(self):
        return self.world

    # control API
    def set_target_position(self, physics: mjcf.Physics, target_position: np.ndarray):
        """set the mocap position"""
        # the bind function is for convenient access to the mujoco mData class that contains all
        # pysics.bind(pointmass.mocap).pos = target_position
        # DO NOT USE THAT here: it returns a view on the physics state, it does not SET the physics state
        # dynamic variables of the simulation
        physics.data.mocap_pos = target_position


if __name__ == "__main__":
    from mujoco_sim.models.utils import create_dummy_arena, write_xml

    arena = create_dummy_arena()
    pointmass = PointMass(parent=arena)
    model = pointmass.mjcf_model
    write_xml(model)

    physics = mjcf.Physics.from_mjcf_model(model)
    for _ in range(1000):  # default timestep  = 2ms -> 40ms.
        # set the mocap target position
        pointmass.set_target_position(physics, np.array([0.10, 0, 0.02]))
        print(f"{physics.named.data.xpos}")
        print(physics.bind(pointmass.mocap).xpos)
        physics.step()
