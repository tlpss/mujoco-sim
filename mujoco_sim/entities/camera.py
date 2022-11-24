import dataclasses

import numpy as np
from dm_control import composer

from dm_control import mjcf
from dm_control.mujoco import Physics
from dm_control.composer.observation.observable import MJCFCamera

@dataclasses.dataclass
class CameraConfig:
    # TODO: bring orientation to the conventional Frame
    #  Orientation is now in MuJoCO frame, where camera points down to the -z axis!
    position: np.ndarray = None
    orientation: np.ndarray = None
    fov: float = None
    image_height: int = 64
    image_width: int = 64


class Camera(composer.Entity):
    @property
    def mjcf_model(self):
        return self._model

    def __init__(self, config: CameraConfig):
        self.config = config
        super().__init__()

    def _build(self):
        self._model = mjcf.RootElement("Camera")
        self._camera = self._model.worldbody.add("camera",mode="fixed",pos=self.config.position, quat=self.config.orientation,fovy=self.config.fov)


    def get_image(self, physics: Physics):
        pixels = physics.render(
            height=self.config.image_height,
            width=self.config.image_width,
            camera_id=self._camera.full_identifier,
            depth=False,
            segmentation=False)
        return pixels

    def get_depth_image(self, physics: Physics):
        depth_map = physics.render(
            height=self.config.image_height,
            width=self.config.image_width,
            camera_id=self._camera.full_identifier,
            depth=False,
            segmentation=False)
        return depth_map
