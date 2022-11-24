from abc import ABC
from typing import Union

from dm_control import composer
from dm_control.mujoco import Physics


class CachedPhysicsEntity(composer.Entity, ABC):
    """
    This class stores the pointer to the Physics element
    so that the methods in the class do not require passing the physics
    element from the task to the entity all the time (unless they are being called
    before the `initialize_episode` function, such as for Observables.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.physics: Union[None, Physics] = None

    def initialize_episode_mjcf(self, random_state):
        self.physics = None

    def initialize_episode(self, physics, random_state):
        self.physics = physics


class Prop(composer.Entity, ABC):
    """
    Prop base class
    """
