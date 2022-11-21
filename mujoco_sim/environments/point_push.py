import dataclasses

import numpy as np
from dm_control import composer
from dm_env import specs

from mujoco_sim.models.point_mass import PointMass2D, build_mocap
from mujoco_sim.models.utils import write_xml
from mujoco_sim.models.walled_arena import WalledArena

REWARD_TYPES = ("shaped", "sparse")
OBSERVATION_TYPES = ("state", "visual")


@dataclasses.dataclass
class Config:
    reward: str = "shaped"
    observation_space = "state"
    max_step_size: float = 0.02
    physics_timestep: float = 0.002  # MJC default
    control_timestep: float = 0.004


class PointMassReachTask(composer.Task):
    """
    Planar pointmass reach task.

    This also served as task to learn dm_control and mujoco, and is therefore 'over'documented.

    ----
    Task implements the environment 'logic' and has a reference
    to the Arena, which is the root of the Entity tree that makes up the physics scene.

    it should implement the

    `root_entity` property, which returns the Arena
    `get_reward` function
    `should_terminate_episode` function

    and can optionally provide
    `task_observables` property to provide observables on top of the observables in the entities
    and many callbacks as described in section 5.3 of the dm_control paper.

    The task will later be wrapped by an 'Environment' to create the RL env
    """

    def __init__(self, config: Config = None) -> None:
        # arena is the convention name for the root of the Entity tree
        self._arena = WalledArena()

        # have to define the mocap here to make sure it is a child of the worldboddy...
        mocap = build_mocap(self._arena.mjcf_model, "pointmass_mocap")
        self.pointmass = PointMass2D(mocap=mocap)
        self._arena.attach(self.pointmass)
        # after attaching, the name has changed but the name of the MJCF elements is not changed?? -> add the prefix manually for the weld.
        self._arena.mjcf_model.equality.add(
            "weld",
            name="mocap_to_mass_weld",
            body1=mocap.name,
            body2=f"{PointMass2D._ROOT_ELEMENT_NAME}/{self.pointmass.pointmass.name}",
        )

        self.target = self._arena.mjcf_model.worldbody.add(
            "site", name="target", type="box", rgba=[0, 255, 0, 1.0], size=[0.01, 0.01, 0.01], pos=[0.25, 0.25, 0.01]
        )
        write_xml(self._arena.mjcf_model)

        self.distance_to_target = 1.0
        self.config = Config() if config is None else config

        # set timesteps
        self.physics_timestep = self.config.physics_timestep
        self.control_timestep = self.config.control_timestep

    def initialize_episode(self, physics, random_state):

        # TODO: should this happen here or in the initialize MJCF?
        # TODO: seems to have no effect?
        # set random target position
        physics.bind(self.target).pos = np.array([[0.25, -0.4, 0.01]])
        # set random pointmass start position
        physics.bind(self.pointmass.pointmass).pos = np.array([0.10, 0.0, self.pointmass.radius])
        physics.bind(self.pointmass.mocap).pos = np.array([0.10, 0.0, self.pointmass.radius])

    @property
    def root_entity(self):
        return self._arena

    def before_step(self, physics, action, random_state):
        # default action is to just map action to the physics.ctrl
        # but if you override action_spec, this probably won't work
        # super().before_step(physics, action, random_state)
        assert action.shape == (2,)

        # get current position.
        current_position = self.pointmass.get_position(physics)
        target_position = current_position + action
        self.pointmass.set_target_position(physics, target_position)

    def after_step(self, physics, random_state):
        self.distance_to_target = np.linalg.norm(
            self.pointmass.global_vector_to_local_frame(physics, physics.bind(self.target).pos)
        )

    def get_reward(self, physics):
        return -self.distance_to_target

    # TODO:
    # should terminate and get_discount (to have termination vs truncation)
    def action_spec(self, physics):
        # if the action space matches the 'actuation space', dm_control will handle this by just broadcasting all the actuators
        # TODO: take super class action space and extend
        bound = np.array([self.config.max_step_size, self.config.max_step_size])
        return specs.BoundedArray((2,), np.float32, -bound, bound)


def create_random_policy(task: composer.Task):
    spec = environment.action_spec()

    def random_policy(time_step):
        return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

    return random_policy


if __name__ == "__main__":
    from dm_control import viewer
    from dm_control.composer import Environment

    task = PointMassReachTask()
    environment = Environment(task)
    print(environment.reset())
    print(environment.action_spec())
    print(environment.step(np.ones(2) * 0.8))
    # TODO: figure out if you can make env render more frequent than control frequency
    # to see intermediate physics.
    # but I'm guessing you can't..
    viewer.launch(environment, policy=create_random_policy(task))
