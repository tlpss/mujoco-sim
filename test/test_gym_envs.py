import gymnasium
import pytest
import numpy as np
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



@pytest.mark.parametrize("env_id", [key for key in gymnasium.registry.keys() if "mujoco_sim" in key])
def test_determinism_of_env(env_id):
    env = gymnasium.make(env_id)
    env.seed(2025)
    obs, _ = env.reset()
    
    env.seed(2025)
    obs2, _ = env.reset()

    for key, value in obs.items():
        assert np.allclose(value, obs2[key], atol=1e-6)

    env.seed(2024)
    obs3, _ = env.reset()
    for key, value in obs.items():
        assert not np.allclose(value, obs3[key], atol=1e-6)


