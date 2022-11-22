import dataclasses

import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from dm_env import specs

from mujoco_sim.models.point_mass import PointMass2D, build_mocap
from mujoco_sim.models.walled_arena import WalledArena

SPARSE_REWARD = "sparse_reward"
DENSE_POTENTIAL_REWARD = "dense_reward"

REWARD_TYPES = (SPARSE_REWARD, DENSE_POTENTIAL_REWARD)
OBSERVATION_TYPES = ("state", "visual")


@dataclasses.dataclass
class Config:
    reward: str = DENSE_POTENTIAL_REWARD
    observation_space = "state"
    max_step_size: float = 0.5
    physics_timestep: float = 0.002  # MJC default
    control_timestep: float = 0.06  # 30 physics steps
    max_control_steps: int = 50

    goal_distance_threshold: float = 0.02  # task solved if dst(point,goal) <


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

        # after attaching, the name has changed, so we can only weld here...
        self._arena.mjcf_model.equality.add(
            "weld",
            name="mocap_to_mass_weld",
            body1=mocap.name,
            body2=self.pointmass.pointmass.full_identifier,  # full identifier, not name!
        )

        self.target = self._arena.mjcf_model.worldbody.add(
            "site", name="target", type="box", rgba=[0, 255, 0, 1.0], size=[0.01, 0.01, 0.01], pos=[0.25, 0.25, 0.01]
        )

        self.config = Config() if config is None else config

        # set timesteps
        self.physics_timestep = self.config.physics_timestep
        self.control_timestep = self.config.control_timestep

        # configure observables of all entities
        self.pointmass.observables.position.enabled = True

        self.goal_position_observable = observable.Generic(self._goal_position)
        self.goal_position_observable.enabled = True

        # initialize some variables for reward calculation etc.
        self.distance_to_target = 1.0
        self.previous_distance_to_target = 1.0

    def initialize_episode(self, physics, random_state: np.random):

        # pose randomisations should happen here.
        #  Varying number of objects should probably happen in the initialize_episode_mjcf
        # set random target position
        goal_x = random_state.uniform(
            self._arena.X_RANGE[0] + self.pointmass.radius, self._arena.X_RANGE[1] - self.pointmass.radius
        )
        goal_y = random_state.uniform(
            self._arena.Y_RANGE[0] + self.pointmass.radius, self._arena.Y_RANGE[1] - self.pointmass.radius
        )
        physics.bind(self.target).pos = np.array([goal_x, goal_y, self.pointmass.radius / 2])
        # set random pointmass start position
        point_initial_x = random_state.uniform(
            self._arena.X_RANGE[0] + self.pointmass.radius, self._arena.X_RANGE[1] - self.pointmass.radius
        )
        point_initial_y = random_state.uniform(
            self._arena.Y_RANGE[0] + self.pointmass.radius, self._arena.Y_RANGE[1] - self.pointmass.radius
        )
        self.pointmass.reset_position(physics, np.array([point_initial_x, point_initial_y]))

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
        target_position = np.clip(target_position, self._arena.X_RANGE[0], self._arena.X_RANGE[1])
        self.pointmass.set_target_position(physics, target_position)

    def after_step(self, physics, random_state):
        # update metrics for reward & termination
        self.previous_distance_to_target = np.copy(self.distance_to_target)
        self.distance_to_target = np.linalg.norm(
            self.pointmass.get_position(physics) - physics.bind(self.target).pos[:2]
        )

    # print(f"{self.pointmass.get_position(physics)} -> {self._goal_position(physics)}")
    def get_reward(self, physics):
        del physics  # unused

        if self.config.reward == SPARSE_REWARD:
            return self.distance_to_target < self.config.goal_distance_threshold
        elif self.config.reward == DENSE_POTENTIAL_REWARD:
            # potential shaped reward
            return -self.distance_to_target

    def _max_time_exceeded(self, physics):
        return physics.data.time > self.config.max_control_steps * self.config.control_timestep

    def should_terminate_episode(self, physics):

        time_limit_reached = self._max_time_exceeded(physics)
        goal_reached = self.distance_to_target < self.config.goal_distance_threshold
        done = time_limit_reached or goal_reached
        return done

    def get_discount(self, physics):
        if self.distance_to_target < self.config.goal_distance_threshold:
            return 0.0  # true terminal state
        return 1.0

    def action_spec(self, physics):
        del physics
        # if the action space matches the 'actuation space', dm_control will handle this by just broadcasting all the actuators
        # TODO: take super class action space and extend
        bound = np.array([self.config.max_step_size, self.config.max_step_size])
        return specs.BoundedArray((2,), np.float32, -bound, bound)

    def _goal_position(self, physics) -> np.ndarray:
        return physics.bind(self.target).pos[:2]

    @property
    def task_observables(self):
        return {"goal_position": self.goal_position_observable}


# TODO: add observations to pointmass
# TODO: create a separate entity for the goals and add observable to it.
# combine them in the env

# TODO: add camera (if pixel obs.)

# How to represent mix of discrete and continuous actions in the replay buffer?

# TODO: demonstration policy -> returns action when given a
# How to convert to gym tuples (s,a,reward, done, info)?
# and how to represent the observations?


# state, proprio = 1D array
# image = 3D array
# tactile sensor can be 1D or 2D
# combinations -> dict?


def create_random_policy(environment: composer.Environment):
    spec = environment.action_spec()
    environment.observation_spec()

    def random_policy(time_step):
        return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

    return random_policy


if __name__ == "__main__":
    from dm_control import viewer
    from dm_control.composer import Environment

    task = PointMassReachTask()
    environment = Environment(task, strip_singleton_obs_buffer_dim=True)
    print(environment.reset())
    print(environment.action_spec())
    print(environment.observation_spec())
    print(environment.step(np.ones(2) * 0.8))
    # TODO: figure out if you can make env render more frequent than control frequency
    # to see intermediate physics.
    # but I'm guessing you can't..
    viewer.launch(environment, policy=create_random_policy(environment))
