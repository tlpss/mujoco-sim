import stable_baselines3.common.policies
from dm_control.composer import Environment
from stable_baselines3.sac.sac import SAC

from mujoco_sim.environments.dmc2gym import DMCWrapper
from mujoco_sim.environments.tasks.point_reach import PointMassReachTask, PointReachConfig
import numpy as np
if __name__ == "__main__":
    from dm_control import viewer
    config = PointReachConfig(observation_type="visual_observations")
    dmc_env = Environment(PointMassReachTask(config),strip_singleton_obs_buffer_dim=True)
    print(dmc_env.observation_spec())
    gym_env = DMCWrapper(dmc_env,flatten_observation_space=False)
    print(gym_env.action_space)
    print(gym_env.observation_space)


    sac = SAC("MultiInputPolicy", env=gym_env, verbose=1, learning_rate=1e-4)
    sac.learn(10000, progress_bar=True)

    def policy(time_step):
        action = sac.predict(gym_env._get_obs(time_step),deterministic=False)[0]
        action = np.array(action)
        return action

    viewer.launch(dmc_env,policy)
