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
