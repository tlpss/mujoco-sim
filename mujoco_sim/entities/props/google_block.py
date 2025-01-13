import random
from typing import List, Tuple

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable

from mujoco_sim.entities.utils import get_assets_root_folder

CATEGORIES = ("cube", "moon", "pentagon", "star")

RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
YELLOW = (1.0, 1.0, 0.0, 1.0)
ORANGE = (1.0, 0.5, 0.0, 1.0)
PURPLE = (1.0, 0.0, 1.0, 1.0)

COLORS = (RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE)


class GoogleBlockProp(composer.Entity):
    XML_OBJECT_NAME = "model"

    def __init__(
        self, block_category: str = "cube", scale: float = 1.0, color: Tuple[float, float, float, float] = RED
    ):
        self.block_category = block_category
        self.scale = scale
        self.color = color
        self.mass = 0.1  # kg
        super().__init__()

    def initialize_episode_mjcf(self, random_state):
        return super().initialize_episode_mjcf(random_state)

    def _build(self):
        self._model = mjcf.from_path(
            str(get_assets_root_folder() / "google_language_table_blocks" / f"{self.block_category}.xml")
        )
        self.object_body = self._model.find("body", self.XML_OBJECT_NAME)

        self._model.find_all("geom")[0].mass = self.mass
        self._model.find_all("geom")[0].rgba = np.array(self.color)
        self._model.find_all("mesh")[0].scale = np.array([self.scale, self.scale, self.scale])

        # try to avoid 'rotation' of the moons by adding rotational friction
        self._model.find_all("geom")[0].condim = 4
        self._model.find_all("geom")[0].friction = np.array([1.0, 0.05, 0.0])

    @property
    def mjcf_model(self):
        return self._model

    @classmethod
    def sample_random_object(
        cls,
        category_list: List[str] = None,
        color_list: List[Tuple] = None,
        scale_range: Tuple[float, float] = (0.8, 1.2),
    ):
        category_lists = category_list or CATEGORIES
        color_list = color_list or COLORS
        block_category = random.choice(category_lists)

        color = random.choice(color_list)
        scale = random.uniform(*scale_range)
        return GoogleBlockProp(block_category, scale, color)

    def get_position(self, physics: mjcf.Physics):
        # do not use pos, but xpos instead
        return physics.bind(self.object_body).xpos

    def _build_observables(self):
        return GoogleBlockObservables(self)


class GoogleBlockObservables(composer.Observables):
    @composer.observable
    def position(self):
        return observable.Generic(self._entity.get_position)


if __name__ == "__main__":
    pass

    block = GoogleBlockProp.sample_random_object(category_list=["cube"])
    model = block.mjcf_model
    physics = mjcf.Physics.from_mjcf_model(model)
    print(block.get_position(physics))

    # visualize mujoco scene

    from dm_control import viewer

    viewer.launch(block.mjcf_model)
