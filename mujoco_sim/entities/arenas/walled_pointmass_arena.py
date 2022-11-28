from dm_control import composer, mjcf

from mujoco_sim.entities.utils import get_assets_root_folder

WALLED_ARENA_XML = "walled_pointmass_arena.xml"


class WalledPointmassArena(composer.Entity):
    X_RANGE = (-0.5, 0.5)
    Y_RANGE = X_RANGE
    """An empty walled 0.5x0.5m arena for PointMass environments"""

    def __init__(self, radius: float = 0.02, mass: float = 0.1) -> None:
        """
        parent: the xml tree is built top-down, to make sure that you always have acces to entire scene so far,
        including the worldbody (for creating mocaps, which have to be children of the worldbody)
        or can define equalities with other elements.
        """
        self.mass = mass
        self.radius = radius
        self.model = None

        # call the super init, which handles the building
        super().__init__()

    def _build(self):
        self.model = mjcf.from_path(str(get_assets_root_folder() / WALLED_ARENA_XML))

    @property
    def mjcf_model(self):
        return self.model


if __name__ == "__main__":
    WalledPointmassArena()
