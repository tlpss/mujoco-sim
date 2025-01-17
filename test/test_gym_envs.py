import gymnasium
import pytest

import mujoco_sim


@pytest.mark.render
@pytest.mark.parametrize("env_id", [key for key in gymnasium.registry.keys() if "mujoco_sim" in key])
def test_env_w_random_policy(env_id):
    print(mujoco_sim)
    env = gymnasium.make(env_id)
    done = False
    obs = env.reset()
    action_policy = env.unwrapped._env.task.create_random_policy()
    while not done:
        obs, reward, term, trunc, info = env.step(action_policy(obs))
        done = term or trunc
