from dm_control.composer import Environment
from stable_baselines3.sac.sac import SAC

from mujoco_sim.environments.dmc2gym import DMCWrapper
from mujoco_sim.environments.tasks.point_reach import PointMassReachTask, PointReachConfig

if __name__ == "__main__":

    config = PointReachConfig(reward="sparse")
    dmc_env = Environment(PointMassReachTask())
    gym_env = DMCWrapper(dmc_env)
    print(gym_env.action_space)
    sac = SAC("MlpPolicy", env=gym_env, verbose=1, learning_rate=1e-3)
    sac.learn(10000, progress_bar=True)

    # viewer.launch(sac,policy)
