from dm_control import composer
from dm_control import mjcf
from mujoco_sim.entities.utils import get_assets_root_folder
from typing import Tuple
TYPES = ("cube","moon","pentagon","star")

RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
YELLOW = (1.0, 1.0, 0.0, 1.0)
ORANGE = (1.0, 0.5, 0.0, 1.0)
PURPLE = (1.0, 0.0, 1.0, 1.0)

COLORS = (RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE)

class GoogleBlockProp(composer.Entity):
    def __init__(self, type: str = "cube", scale:float = 1.0, color: Tuple[float,float,float,float]= RED):
        self.type = type
        self.scale = scale
        self.color = color
        self.mass = 0.1 # kg
        super().__init__()
    
    def _build(self):
        self._model = mjcf.RootElement()
        self._model.get_assets().add("asset",type="mesh",file=str(get_assets_root_folder() / "google_language_table_blocks" / f"{self.type}.obj"))
        self.block = self._model.worldbody.add(
            "geom",
            type="mesh",
            mesh=self.type,
            rgba=[0.2, 0.2, 0.2, 1.0],
            pos=[0.0, 0.0, 0.0],
            mass=self.mass,
        )

    @property
    def mjcf_model(self):
        return self._model
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    block = GoogleBlockProp()
    model = block.mjcf_model
    physics = mjcf.Physics.from_mjcf_model(model)
    
    