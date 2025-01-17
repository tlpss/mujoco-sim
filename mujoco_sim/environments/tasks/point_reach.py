import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_env import specs

from mujoco_sim.entities.arenas import WalledPointmassArena
from mujoco_sim.entities.camera import Camera, CameraConfig
from mujoco_sim.entities.pointmass import PointMass2D
from mujoco_sim.entities.utils import build_mocap

SPARSE_REWARD = "sparse_reward"
DENSE_POTENTIAL_REWARD = "dense_potential_reward"
DENSE_NEG_DISTANCE_REWARD = "dense_negative_distance_reward"
DENSE_BIASED_NEG_DISTANCE_REWARD = "dense_biased_negative_distance_reward"

STATE_OBS = "state_observations"
VISUAL_OBS = "visual_observations"

REWARD_TYPES = (SPARSE_REWARD, DENSE_POTENTIAL_REWARD, DENSE_NEG_DISTANCE_REWARD, DENSE_BIASED_NEG_DISTANCE_REWARD)
OBSERVATION_TYPES = (STATE_OBS, VISUAL_OBS)

TOP_DOWN_CAMERA_CONFIG = CameraConfig(np.array([0.0, 0.0, 2.4]), np.array([1.0, 0.0, 0.0, 0.0]), 30)

PHYSICS_TIMESTEP = 0.02  # MJC default =0.002
CONTROL_TIMESTEP = 0.1
MAX_CONTROL_STEPS_PER_EPISODE = 50
GOAL_DISTANCE_THRESHOLD = 0.02
MAX_STEP_SIZE = 0.05


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


    cf. Kevin Zakka's RoboPianist for a well-coded example of how to create RL envs with dm_control
    https://github.com/google-research/robopianist/tree/main/robopianist
    """

    MAX_CONTROL_STEPS_PER_EPISODE = MAX_CONTROL_STEPS_PER_EPISODE
    CONTROL_TIMESTEP = CONTROL_TIMESTEP

    def __init__(
        self,
        reward_type: str = DENSE_BIASED_NEG_DISTANCE_REWARD,
        observation_type: str = VISUAL_OBS,
        image_resolution: int = 64,
    ) -> None:
        super().__init__()
        assert reward_type in REWARD_TYPES
        assert observation_type in OBSERVATION_TYPES
        self.reward_type = reward_type
        self.observation_type = observation_type
        self.image_resolution = image_resolution

        # arena is the convention name for the root of the Entity tree
        self._arena = WalledPointmassArena()

        # have to define the mocap here to make sure it is a child of the worldboddy...
        # mocap serves as 'actuator' for the pointmass
        mocap = build_mocap(self._arena.mjcf_model, "pointmass_mocap")
        self.pointmass = PointMass2D(mocap=mocap, radius=0.05)
        self._arena.attach(self.pointmass)

        # after attaching, the name has changed, so we can only weld here...
        self._arena.mjcf_model.equality.add(
            "weld",
            name="mocap_to_mass_weld",
            body1=mocap.name,
            body2=self.pointmass.pointmass.full_identifier,  # full identifier, not name!
        )

        self.target = self._arena.mjcf_model.worldbody.add(
            "site", name="target", type="box", rgba=[0, 255, 0, 1.0], size=[0.04, 0.04, 0.04], pos=[0.25, 0.25, 0.01]
        )
        # add Camera to scene
        top_down_config = TOP_DOWN_CAMERA_CONFIG
        top_down_config.image_width = top_down_config.image_height = self.image_resolution
        self.camera = Camera(top_down_config)
        self._arena.attach(self.camera)

        # set timesteps
        self.physics_timestep = PHYSICS_TIMESTEP
        self.control_timestep = CONTROL_TIMESTEP

        # create additional observables / Sensors
        self.goal_position_observable = observable.Generic(self._goal_position)
        self._task_observables = {
            "goal_position": self.goal_position_observable,
        }
        self._configure_observables()

        # initialize some variables for reward calculation etc.
        self.distance_to_target = 1.0
        self.previous_distance_to_target = 1.0

    def _configure_observables(self):
        if self.observation_type == STATE_OBS:
            self.pointmass.observables.position.enabled = True
            self.goal_position_observable.enabled = True
        elif self.observation_type == VISUAL_OBS:
            self.camera.observables.rgb_image.enabled = True
            self.pointmass.observables.position.enabled = True
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

    def get_reward(self, physics):
        del physics  # unused

        if self.reward_type == SPARSE_REWARD:
            return self.distance_to_target < self.config.goal_distance_threshold
        elif self.reward_type == DENSE_NEG_DISTANCE_REWARD:
            return -self.distance_to_target
        elif self.reward_type == DENSE_BIASED_NEG_DISTANCE_REWARD:
            # TODO: make this smarter by doing 1-tanh(b*distance) which is always positive.
            # and configure it in such a way that the total reward / step is <=1 with 1 -> succesful termination?
            return -self.distance_to_target + 0.5  # attempt to make random policy reward positive
        elif self.reward_type == DENSE_POTENTIAL_REWARD:
            return self.previous_distance_to_target - self.distance_to_target

        else:
            raise ValueError("reward type not known?")

    def should_terminate_episode(self, physics):
        # time limit violations (truncations) for finite formulations if infinite horizon task are not handled here!
        # they are handled in the environment (using the MAX_CONTROL_STEPS_PER_EPISODE)
        # only terminal states (goal reached or invalid states, e.g. collisions)
        return self.is_goal_reached(physics)

    def is_goal_reached(self, physics):
        return self.distance_to_target < GOAL_DISTANCE_THRESHOLD

    def get_discount(self, physics):
        # feature of DM env that is not used in Gymnasium Env..
        if self.is_goal_reached(physics):
            return 0.0
        return 1.0

    def action_spec(self, physics):
        del physics
        # if the action space matches the 'actuation space', dm_control will handle this by just broadcasting all the actuators
        # TODO: take super class action space and extend
        bound = np.array([MAX_STEP_SIZE, MAX_STEP_SIZE])
        return specs.BoundedArray((2,), np.float32, -bound, bound)

    def _goal_position(self, physics) -> np.ndarray:
        return physics.bind(self.target).pos[:2]

    @property
    def task_observables(self):
        return {name: obs for (name, obs) in self._task_observables.items() if obs.enabled}

    def create_random_policy(self):
        physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        spec = self.action_spec(physics)

        def random_policy(time_step):
            return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

        return random_policy

    def create_demonstation_policy(self, environment, noise: float = 0.0):
        def policy(time_step):
            physics = environment.physics
            current_position = self.pointmass.get_position(physics)
            target_position = self._goal_position(physics)

            action = target_position - current_position
            if noise > 0:
                action *= 1 + np.random.normal(0, noise, action.shape)
            action *= 1 / np.max(np.abs(action)) * MAX_STEP_SIZE

            return action

        return policy


if __name__ == "__main__":
    from dm_control import viewer
    from dm_control.composer import Environment

    task = PointMassReachTask(observation_type=STATE_OBS)
    environment = Environment(task, strip_singleton_obs_buffer_dim=True)
    timestep = environment.reset()
    print(environment.action_spec())
    print(environment.observation_spec())
    # print(environment.step(None))
    # write_xml(task._arena.mjcf_model)
    img = task.camera.get_rgb_image(environment.physics)

    # import matplotlib.pyplot as plt
    # plt.imsave("test.png", timestep.observation["Camera/rgb_image"])

    viewer.launch(environment, policy=task.create_demonstation_policy(environment, noise=0))
