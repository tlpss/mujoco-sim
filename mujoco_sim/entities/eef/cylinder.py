import abc

import numpy as np
from dm_control import composer, mjcf

from mujoco_sim.entities.utils import write_xml
from mujoco_sim.type_aliases import VECTOR_TYPE


class EEF(composer.Entity, abc.ABC):
    @property
    def tcp_offset(self) -> VECTOR_TYPE:
        """the offset of the TCP in the robot flange frame"""
        raise NotImplementedError


class CylinderEEF(EEF):
    def __init__(self, len: float = 0.1, radius: float = 0.02):
        self.len = len
        self.radius = radius
        super().__init__()

    def _build(self):
        self._model = mjcf.RootElement()
        self.cylinder = self._model.worldbody.add(
            "geom",
            name="cylinder-EEF",
            type="cylinder",
            mass=0.1,
            size=[self.radius, self.len / 2],
            rgba=[0.2, 0.2, 0.2, 1.0],
            pos=[0.0, 0.0, +self.len / 2 + 0.001],
        )

    @property
    def mjcf_model(self):
        return self._model

    @property
    def tcp_offset(self) -> VECTOR_TYPE:
        return np.array([0.0, 0.0, self.len])


if __name__ == "__main__":
    cylinder = CylinderEEF()
    write_xml(cylinder.mjcf_model)
