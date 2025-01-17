""" Adapter for the dm_env to create a gymnasium Env Interface

based on  https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py

DMC env api documentation: https://github.com/google-deepmind/dm_env/blob/master/docs/index.md, https://arxiv.org/pdf/2006.12983#page=11.68
gym env api documentation: https://gymnasium.farama.org/api/env/
"""

from typing import List

import gymnasium
import numpy as np
from dm_control.composer import Environment
from dm_env import specs
from gymnasium import spaces


def _convert_specs_to_flattened_box(spec: List[specs.Array], dtype: np.dtype) -> spaces.Box:
    """Converts a list of n-dimensional specs to a gymnasium.spaces.Box
    by flattening each spec and then concatenating the 1D-sizes.

    e.g. (2,),(2,2) spec becomes (6,) box

    Args:
        spec (List[specs.Array]): _description_
        dtype (np.dtype): _description_

    Returns:
        spaces.Box: _description_
    """

    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int32(np.prod(s.shape))  # np.numel would be cleaner?
        if isinstance(s, specs.Array):
            bound = np.inf * np.ones(dim, dtype=dtype)
            return -bound, bound
        elif isinstance(s, specs.BoundedArray):
            zeros = np.zeros(dim, dtype=dtype)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape

    # cast to float32 to save some space in replay buffers?
    return spaces.Box(low, high, dtype=np.float32)


def convert_spec_to_box(s):
    dim = s.shape
    if isinstance(s, specs.Array):
        bound = np.inf * np.ones(dim, dtype=s.dtype)
        low, high = -bound, bound
    if isinstance(s, specs.BoundedArray):
        zeros = np.zeros(dim, dtype=s.dtype)
        low, high = s.minimum + zeros, s.maximum + zeros
    return spaces.Box(low, high, dtype=s.dtype)


def _flatten_obs(obs: dict) -> np.ndarray:
    """takes a dict of {"str": np.ndarray} observations (according to spec)
    and returns a flattened np.array

    """
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCEnvironmentAdapter(gymnasium.Env):
    def __init__(
        self,
        env: Environment,
        flatten_observation_space: bool = False,
        render_camera_id=-1,  # the free camera that is always available.
        render_dims: tuple[int, int] = (256, 256),
    ):
        self.flatten_observation_space = flatten_observation_space
        self._camera_id = render_camera_id
        self.render_dims = render_dims
        self._env = env
        self._action_space = _convert_specs_to_flattened_box([self._env.action_spec()], np.float32)

        # create observation space
        if flatten_observation_space:
            self._observation_space = _convert_specs_to_flattened_box(
                self._env.observation_spec().values(), np.float64
            )

        else:
            self._observation_space = spaces.Dict(
                {key: convert_spec_to_box(spec) for key, spec in self._env.observation_spec().items()}
            )
            # idea is to maintain the dict structure here and simply turn the dict of specs into a dict of boxes
            # this way dimensionality is not lost (e.g. F/T sensor and image can exist w/o having to flatten the image)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        obs = time_step.observation
        if self.flatten_observation_space:
            obs = _flatten_obs(obs)
        return obs

    @property
    def dmc_env(self):
        return self._env

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed):
        # set the seed of the random generator of the episode
        # will be applied during reset by the DMC env
        self._env._fixed_random_state = np.random.RandomState(seed)
        self._action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        info = {}

        time_step = self._env.step(action)
        reward = time_step.reward
        obs = self._get_obs(time_step)
        # truncated = finite-horizon formulation timeout of infinite task
        # env signals stop but agent should bootstrap from next_obs so discount > 0
        # as this is not a true terminal state
        # cf https://arxiv.org/pdf/1712.00378.pdf

        truncated = time_step.last() and time_step.discount > 0
        terminated = time_step.last() and time_step.discount == 0

        # check for success and add it toinfo dict, to please Lerobot which expects the 'is_success' key in the info dict
        # check if method "is_goal_reached" exists in the env
        if hasattr(self._env.task, "is_goal_reached"):
            info["is_success"] = self._env.task.is_goal_reached(self._env.physics) * 1.0

        # log discount as it is not passed explicitly in gym env
        info["discount"] = time_step.discount

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            self.seed(seed)
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        info = {}  # new gym API requires info in reset as well.
        return obs, info

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height, width = self.render_dims
        return self._env.physics.render(height=height, width=width, camera_id=self._camera_id)


if __name__ == "__main__":
    # from dm_control import suite

    # env = suite.load(domain_name="cartpole", task_name="swingup")

    from dm_control.composer import Environment

    from mujoco_sim.environments.tasks.point_reach import PointMassReachTask

    task = PointMassReachTask()
    env = Environment(
        task,
        strip_singleton_obs_buffer_dim=True,
        time_limit=PointMassReachTask.MAX_CONTROL_STEPS_PER_EPISODE * PointMassReachTask.CONTROL_TIMESTEP,
    )
    print(env.action_spec())
    print(env.observation_spec())
    gym_env = DMCEnvironmentAdapter(env, flatten_observation_space=False, render_dims=(256, 256))
    print(f"{gym_env.observation_space=}")
    print(f"{gym_env.action_space=}")
    obs, info = gym_env.reset()
    print(f"{obs=}")
    obs, reward, terminated, truncated, info = gym_env.step(gym_env.action_space.sample())
    done = terminated or truncated
    img = gym_env.render()
    print(img.shape)
    print(f"{obs=}")
    print(f"{reward=}")
    print(f"{info=}")

    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()

    print(env._time_limit)
    policy = env.task.create_random_policy()
    done = False
    obs = gym_env.reset()
    while not done:
        obs, reward, term, trunc, info = gym_env.step(policy(obs))
        done = term or trunc
    print(info)
    print(gym_env.dmc_env.task.is_goal_reached(gym_env.dmc_env.physics))
