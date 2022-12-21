from typing import Tuple

import numpy as np


class EuclideanSpace:
    def __init__(
        self, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float]
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def clip_to_space(self, pose: np.ndarray) -> np.ndarray:
        for i, axis in enumerate([self.x_range, self.y_range, self.z_range]):
            pose[i] = np.clip(pose[i], axis[0], axis[1])
        return pose

    def is_in_space(self, pose: np.ndarray) -> bool:
        if np.isclose(pose, self.clip_to_space(pose)).all():
            return True
        return False

    def sample(self) -> np.ndarray:
        return np.array(
            [np.random.uniform(*self.x_range), np.random.uniform(*self.y_range), np.random.uniform(*self.z_range)]
        )

    def create_visualization_site(self, worldbody, name: str):
        x, y, z = sum(self.x_range) / 2, sum(self.y_range) / 2, sum(self.z_range) / 2
        x_size, y_size, z_size = (
            self.x_range[1] - self.x_range[0],
            self.y_range[1] - self.y_range[0],
            self.z_range[1] - self.z_range[0],
        )
        eps = 0.001  # to deal with 2D or 1D spaces
        return worldbody.add(
            "site",
            name=name,
            pos=[x, y, z],
            size=[abs(x_size / 2) + eps, abs(y_size / 2) + eps, abs(z_size / 2) + eps],
            type="box",
            rgba=[1, 1, 1, 0.5],
        )
