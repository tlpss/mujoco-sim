from dm_control import composer, mjcf


class EmptyRobotArena(composer.Entity):
    """An empty arena with a single plane"""

    def __init__(self, plane_side_len: float = 2.0) -> None:
        """ """
        self.plane_length = plane_side_len
        self.model = None

        # call the super init, which handles the building
        super().__init__()

    def _build(self):
        self.arena = mjcf.RootElement()

        self.arena.worldbody.add(
            "geom", type="plane", size=[self.plane_length / 2, self.plane_length / 2, 0.1], rgba=[150, 150, 150, 1]
        )
        for x in [-self.plane_length, self.plane_length, 0.5]:
            self.arena.worldbody.add("light", pos=[x, x, 3])
            self.arena.worldbody.add("light", pos=[x, -x, 3])
        return self.arena

    @property
    def mjcf_model(self):
        return self.model


if __name__ == "__main__":
    EmptyRobotArena()
