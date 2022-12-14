from dm_control import composer, mjcf
from dm_control.composer.observation import observable

from mujoco_sim.entities.utils import get_assets_root_folder, write_xml
from mujoco_sim.type_aliases import VECTOR_TYPE


class CabinetBody(composer.Entity):
    """Parametric body for a cabinet in mjcf.

    defined by height, width, depth, panel_thickness.
    has no door as this is added later

    Args:
        composer (_type_): _description_
    """

    def __init__(
        self, width: float = 0.3, length: float = 0.3, height: float = 0.3, material_thickness: float = 0.01
    ) -> None:
        self.width = width
        self.length = length
        self.height = height
        self.material_thickness = material_thickness

        self._model = None
        super().__init__()

    def _build(self):

        # do not use a predefined mesh, but create procedurally to allow for
        # scaling the cabinet with constant material thickness

        self._model = mjcf.RootElement("cabinet-body")
        bottom_position = [self.width / 2, self.length / 2, self.material_thickness / 2]
        top_side_position = [self.width / 2, self.length / 2, self.height - self.material_thickness / 2]
        left_side_position = [self.material_thickness / 2, self.length / 2, self.height / 2]
        right_side_position = [self.width - self.material_thickness / 2, self.length / 2, self.height / 2]
        back_side_position = [self.width / 2, self.length - self.material_thickness / 2, self.height / 2]

        self.bottom = self._add_panel("bottom", position=bottom_position)
        self.left_side = self._add_panel("left_side", position=left_side_position)
        self.right_side = self._add_panel("right_side", position=right_side_position, size=left_side_position)
        self.top_side = self._add_panel("top_side", position=top_side_position, size=bottom_position)
        self.back_side = self._add_panel(
            "back_side",
            position=back_side_position,
            size=[
                (self.width - self.material_thickness) / 2,
                self.material_thickness / 2,
                (self.height - self.material_thickness) / 2,
            ],
        )

    def _add_panel(self, name: str, position: list, size: list = None):
        size = size or position
        body = self._model.worldbody.add("body", name=name, pos=position)
        body.add("geom", type="box", size=size)
        return body

    @property
    def mjcf_model(self):
        return self._model


class Handle(composer.Entity):
    """Entity for Handle of a cabinet door.
    loads an xml with a mesh of the handle with a body named 'model'
    and a site named 'grasp_site' on which the robot can grasp the handle

    """

    # TODO: dynamically load the asset and the grasp site location, so that there is no need
    # to have a specific xml for each handle (that contains lots of duplication)
    body_name = "model"
    grasp_site_name = "grasp_site"
    asset_path = get_assets_root_folder() / "cabinets" / "handles"

    def __init__(self, id: int = 1):
        assert (
            id <= Handle.get_number_of_handles()
        ), f"Handle with id {id} does not exist. There are only {Handle.get_number_of_handles()} handles available."
        self.id = id
        self._model = None
        super().__init__()

    def _build(self):
        self._model = mjcf.from_path(str(Handle.asset_path / f"handle_{self.id:03d}.xml"))

    @staticmethod
    def get_number_of_handles():
        return len(list((Handle.asset_path).glob("handle_*.xml")))

    @property
    def mjcf_model(self):
        return self._model

    @property
    def grasp_site(self):
        return self._model.worldbody.find("site", Handle.grasp_site_name)

    def _build_observables(self):
        return super()._build_observables()


class HingeCabinet(composer.Entity):
    hinge_axes_dict = {"left": [0, 0, -1], "right": [0, 0, 1], "down": [1, 0, 0], "up": [-1, 0, 0]}

    def __init__(self, cabinet_body: CabinetBody = None, hinge_side: str = "left", handle: Handle = None):
        self.cabinet_body = cabinet_body or CabinetBody()

        assert (
            hinge_side in HingeCabinet.hinge_axes_dict.keys()
        ), f"hinge_side must be one of {HingeCabinet.hinge_axes_dict.keys()}"
        self.hinge_side = hinge_side
        self.handle = handle or Handle()
        self._model = None
        super().__init__()

    def _build(self, *args, **kwargs):
        self._model = mjcf.RootElement("hinge-cabinet")
        self._model.compiler.autolimits = True
        self._model.compiler.angle = "radian"
        # create a body to attach joints to (cannot do that in worldbody)
        self._model.worldbody.attach(self.cabinet_body.mjcf_model)

        # create door and attach with hinge to the cabinet
        self.door = self._model.worldbody.add(
            "body", name="door", pos=[0, -0.001, 0]
        )  # little gap between door and body
        self.door.add(
            "geom",
            type="box",
            size=[self.cabinet_body.width / 2, self.cabinet_body.material_thickness / 2, self.cabinet_body.height / 2],
            pos=[self.cabinet_body.width / 2, -self.cabinet_body.material_thickness / 2, self.cabinet_body.height / 2],
        )
        self.joint = self.door.add(
            "joint", name="door_hinge", type="hinge", limited="true", axis=[0, 0, -1], pos=[0, 0, 0], range=[0, 1.57]
        )
        self.handle_site = self.door.add(
            "site",
            name="handle-site",
            pos=[self.cabinet_body.width * 4 / 5, 0, self.cabinet_body.height / 2],
            size=[0.01],
        )  # create site because cannot attach to body?

        # TODO: add variations to the handle position and orientation
        # TODO: add variations to the range of the hinge
        # TODO: add hinge dynamics
        # TODO: create variations of the hinge side

        # attach the handle
        self.handle_site.attach(self.handle.mjcf_model)

    @property
    def mjcf_model(self):
        return self._model

    def get_grasp_position(self, physics: mjcf.Physics) -> VECTOR_TYPE:
        """returns the grasp position of the handle in the worldbody frame"""
        return physics.named.data.site_xpos[self.handle.grasp_site.full_identifier]

    def _build_observables(self):
        return CabinetObservables(self)


class CabinetObservables(composer.Observables):
    @composer.observable
    def hinge_angle(self) -> observable.Observable:
        return observable.MJCFFeature("qpos", self._entity.joint)


if __name__ == "__main__":

    cabinet = HingeCabinet()
    phsyics = mjcf.Physics.from_mjcf_model(cabinet.mjcf_model)
    with phsyics.reset_context():
        phsyics.data.qpos[0] = 0.5
    print(cabinet.get_grasp_position(phsyics))
    write_xml(cabinet.mjcf_model)
