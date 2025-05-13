import dataclasses
from typing import Tuple

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.mujoco import Physics
from dm_env import specs
import mujoco

def camera_orientation_from_look_at(position: np.ndarray, look_at: np.ndarray, up: np.ndarray):
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)

    rot_matrix = np.array([right, up, forward])
    # convert to quaternion
    quat = np.zeros(4)
    rot_matrix = rot_matrix.flatten()
    mujoco.mju_mat2Quat(quat, rot_matrix)
    return quat

@dataclasses.dataclass
class CameraConfig:
    # TODO: bring orientation to the conventional Frame
    #  Orientation is now in MuJoCO frame, where camera points down to the -z axis!
    position: np.ndarray = None
    orientation: np.ndarray = None
    fov: float = None
    image_height: int = 96
    image_width: int = 96
    name: str = "Camera"


class Camera(composer.Entity):
    @property
    def mjcf_model(self):
        return self._model

    def __init__(self, config: CameraConfig):
        self.config = config
        super().__init__()

    def _build(self, name: str = "Camera"):
        self._model = mjcf.RootElement(self.config.name)
        self.geom = self._model.worldbody.add(
            "geom",
            type="box",
            pos=self.config.position,
            quat=self.config.orientation,
            size=[0.045, 0.0125, 0.0125],
            rgba=[0, 0, 0, 1],
        )
        # self.lens  = self._model.worldbody.add("geom", type="sphere", pos = self.config.position , size=[0.0125,0.0125], rgba=[0, 0, 0, 1])
        self._camera = self._model.worldbody.add(
            "camera", mode="fixed", pos=self.config.position, quat=self.config.orientation, fovy=self.config.fov
        )
        # TODO: add geometric element (for wrist camera's this becomes important.)

    def get_rgb_image(self, physics: Physics):
        # channel-last, 0-255 int
        pixels = physics.render(
            height=self.config.image_height,
            width=self.config.image_width,
            camera_id=self._camera.full_identifier,
            depth=False,
            segmentation=False,
        )
        return pixels

    def get_depth_map(self, physics: Physics):

        depth_map = physics.render(
            height=self.config.image_height,
            width=self.config.image_width,
            camera_id=self._camera.full_identifier,
            depth=False,
            segmentation=False,
        )
        return depth_map

    @property
    def rgb_image_shape(self):
        return self.config.image_height, self.config.image_width, 3

    def _build_observables(self):
        return CameraObservables(self)


class RGBObservable(observable.Generic):
    """
    Observable for RGB image.
    Specifies the array spec to have no confusion on
    channel-first/last and int vs float pixel values.
    """

    def __init__(
        self,
        raw_observation_callable,
        image_shape: Tuple[int, int, int],
        pixels_as_int=True,
        update_interval=1,
        buffer_size=1,
        delay=None,
        aggregator=None,
        corruptor=None,
    ):
        self.image_shape = image_shape
        self.pixels_as_int = pixels_as_int
        super().__init__(raw_observation_callable, update_interval, buffer_size, delay, aggregator, corruptor)

    @property
    def array_spec(self):
        max = 255 if self.pixels_as_int else 1.0
        dtype = np.uint8 if self.pixels_as_int else np.float32
        return specs.BoundedArray(self.image_shape, dtype, 0, max)


class CameraObservables(composer.Observables):

    _entity: Camera  # typing

    @composer.observable
    def rgb_image(self) -> RGBObservable:
        obs = RGBObservable(self._entity.get_rgb_image, self._entity.rgb_image_shape, True)
        return obs

    # TODO: add depth map


if __name__ == "__main__":
    position = np.array([1, 0,0])
    look_at = np.array([0, 0, 0])
    up = np.array([0, 0,1])
    quaternion = camera_orientation_from_look_at(position, look_at, up)
    print(quaternion)