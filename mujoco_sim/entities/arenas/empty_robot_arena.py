from dm_control import composer, mjcf


class EmptyRobotArena(composer.Entity):
    """An empty arena with a single plane"""

    def __init__(self, plane_side_len: float = 2.0) -> None:
        """ """
        self.plane_length = plane_side_len
        self._model = None

        # call the super init, which handles the building
        super().__init__()

    def _build(self):
        self._model = mjcf.RootElement()

        self._model.worldbody.add(
            "geom", type="plane", size=[self.plane_length / 2, self.plane_length / 2, 0.1], rgba=[0.3, 0.3, 0.3, 1]
        )

        self.robot_site = self._model.worldbody.add("site", name="robot_site", pos=[0.0, 0.0, 0])

        for x in [-self.plane_length, self.plane_length, 0.5]:
            self._model.worldbody.add("light", pos=[x, x, 3], castshadow=False)
            self._model.worldbody.add("light", pos=[x, -x, 3], castshadow=False)
        return self._model

    @property
    def mjcf_model(self):
        return self._model

    @property
    def robot_attachment_site(self):
        return self.robot_site


if __name__ == "__main__":
    EmptyRobotArena()
