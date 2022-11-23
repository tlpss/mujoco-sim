""" wrapper for the dm_env to gym Interface

adapted from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
"""

from typing import List

import gym
import numpy as np
from dm_control.composer import Environment
from dm_env import specs
from gym import spaces


def _convert_specs_to_flattened_box(spec: List[specs.Array], dtype: np.dtype) -> spaces.Box:
    """Converts a list of n-dimensional specs to a gym.spaces.Box
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
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
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


def _flatten_obs(obs: dict) -> np.ndarray:
    """takes a dict of {"str": np.ndarray} observations (according to spec)
    and returns a flattened np.array

    """
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(gym.Env):
    def __init__(
        self,
        env: Environment,
        seed: int = 2022,
        flatten_observation_space: bool = True,
        render_camera_id=0,
        render_dims: tuple[int, int] = (96, 96),
    ):
        self.flatten_observation_space = flatten_observation_space
        self._camera_id = render_camera_id
        self.render_dims = render_dims
        self._env = env

        # TODO: check camera exists?

        self._action_space = _convert_specs_to_flattened_box([self._env.action_spec()], np.float32)
        # create observation space
        if flatten_observation_space:
            self._observation_space = _convert_specs_to_flattened_box(
                self._env.observation_spec().values(), np.float64
            )

        else:
            raise NotImplementedError
            # idea is to maintain the dict structure here and simply turn the dict of specs into a dict of boxes
            # this way dimensionality is not lost (e.g. F/T sensor and image can exist w/o having to flatten the image)

        self.current_state = None

        # set seed
        self.seed(seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        obs = time_step.observation
        if self.flatten_observation_space:
            obs = _flatten_obs(obs)
        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed):
        self._action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._action_space.contains(action)
        info = {}

        time_step = self._env.step(action)
        reward = time_step.reward
        obs = self._get_obs(time_step)

        # truncated = finite-horizon formulation timeout of inifite task
        # env signals stop but agent should bootstrap from next_obs
        # as this is not a true terminal state
        # cf https://arxiv.org/pdf/1712.00378.pdf
        truncated = time_step.discount > 0.0 and time_step.last()
        terminated = time_step.last() and time_step.discount == 0.0

        done = truncated or terminated
        info["discount"] = time_step.discount
        info["TimeLimit.truncated"] = truncated
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height, width = self.render_dims
        return self._env.physics.render(height=height, width=width, camera_id=self.render)


if __name__ == "__main__":
    from dm_control import suite

    env = suite.load(domain_name="cartpole", task_name="swingup")
    print(env.action_spec())
    gym_env = DMCWrapper(env)
    print(f"{gym_env.observation_space=}")
    print(f"{gym_env.action_space=}")
    obs = gym_env.reset()
    print(f"{obs=}")
    obs, reward, done, info = gym_env.step(gym_env.action_space.sample())
    # img = gym_env.render()
    print(f"{obs=}")
    print(f"{reward=}")
    print(f"{info=}")