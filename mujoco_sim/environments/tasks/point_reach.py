import dataclasses

import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from dm_env import specs

from mujoco_sim.entities.arenas import WalledPointmassArena
from mujoco_sim.entities.pointmass import PointMass2D
from mujoco_sim.entities.utils import build_mocap, write_xml
from mujoco_sim.environments.tasks.base import TaskConfig
from mujoco_sim.entities.camera import Camera, CameraConfig

SPARSE_REWARD = "sparse_reward"
DENSE_POTENTIAL_REWARD = "dense_potential_reward"
DENSE_NEG_DISTANCE_REWARD = "dense_negative_distance_reward"

STATE_OBS = "state_observations"
VISUAL_OBS = "visual_observations"

REWARD_TYPES = (SPARSE_REWARD, DENSE_POTENTIAL_REWARD, DENSE_NEG_DISTANCE_REWARD)
OBSERVATION_TYPES = (STATE_OBS, VISUAL_OBS)

TOP_DOWN_CAMERA_CONFIG = CameraConfig(np.array([0.0,0.0,1.0]),np.array([1.0,0.0,0.0,0.0]),30)

@dataclasses.dataclass
class PointReachConfig(TaskConfig):
    reward_type: str = DENSE_NEG_DISTANCE_REWARD
    observation_type: str = STATE_OBS
    limit_mocap_workspace: bool = True
    """ limit the mocap (x,y) pose to the walled_arena to, if this is false the mocap is allowed to move outside of the
    arena. The actual ball geom will be stopped by the wall ofc."""
    max_step_size: float = 0.1
    physics_timestep: float = 0.002  # MJC default
    control_timestep: float = 0.06  # 30 physics steps
    max_control_steps_per_episode = 50

    goal_distance_threshold: float = 0.02  # task solved if dst(point,goal) < threshold
    image_resolution: int = 64

    def __post_init__(self):
        assert self.observation_type in OBSERVATION_TYPES
        assert self.reward_type in REWARD_TYPES


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

    def __init__(self, config: PointReachConfig = None) -> None:
        self.config = PointReachConfig() if config is None else config

        # arena is the convention name for the root of the Entity tree
        self._arena = WalledPointmassArena()

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
        # add Camera to scene
        top_down_config = TOP_DOWN_CAMERA_CONFIG
        top_down_config.image_width = top_down_config.image_height = self.config.image_resolution
        self.camera = Camera(top_down_config)
        self._arena.attach(self.camera)

        # set timesteps
        self.physics_timestep = self.config.physics_timestep
        self.control_timestep = self.config.control_timestep

        # create additional observables / Sensors
        self.goal_position_observable = observable.Generic(self._goal_position)
        self.camera_rgb_observable = observable.Generic(self.camera.get_image)
        self._task_observables = {
            "goal_position": self.goal_position_observable,
            "image": self.camera_rgb_observable
        }
        self._configure_observables()

        # initialize some variables for reward calculation etc.
        self.distance_to_target = 1.0
        self.previous_distance_to_target = 1.0

    def _configure_observables(self):
        if self.config.observation_type == STATE_OBS:
            self.pointmass.observables.position.enabled = True
            self.goal_position_observable.enabled = True
        elif self.config.observation_type == VISUAL_OBS:
            self.camera_rgb_observable.enabled = True
        else:
            raise NotImplementedError

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
        if action is None:
            # non-agent call of the step(), most likely to accomplish synchronous actions
            return

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

        if self.config.reward_type == SPARSE_REWARD:
            return self.distance_to_target < self.config.goal_distance_threshold
        elif self.config.reward_type == DENSE_NEG_DISTANCE_REWARD:
            # potential shaped reward
            return -self.distance_to_target
        elif self.config.reward_type == DENSE_POTENTIAL_REWARD:
            return self.previous_distance_to_target - self.distance_to_target

    def _max_time_exceeded(self, physics):
        return physics.data.time > self.config.max_control_steps_per_episode * self.config.control_timestep

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
        return {name: obs for (name,obs) in self._task_observables.items() if obs.enabled}


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

    task = PointMassReachTask(PointReachConfig(observation_type=VISUAL_OBS))
    environment = Environment(task, strip_singleton_obs_buffer_dim=True)
    print(environment.reset())
    print(environment.action_spec())
    print(environment.observation_spec())
    print(environment.step(None))
    write_xml(task._arena.mjcf_model)
    img = task.camera.get_image(environment.physics)
    # TODO: figure out if you can make env render more frequent than control frequency
    # to see intermediate physics.
    # but I'm guessing you can't..
    import matplotlib.pyplot as plt
    plt.imsave("test.png",img)
    viewer.launch(environment, policy=create_random_policy(environment))
