import dataclasses

from dm_control.composer import Environment

from mujoco_sim import _LOGGING_DIR
from mujoco_sim.environments.tasks.point_reach import PointMassReachTask, PointReachConfig
from scripts.sb3_sac import sb3_sac

if __name__ == "__main__":

    @dataclasses.dataclass
    class WandbConfig:
        lr = 3e-4
        tau = 0.005
        gamma = 0.95
        timesteps = 200_000
        seed = 2022
        entropy_coefficient = 0.001
        batch_size = 256

    log_dir = _LOGGING_DIR / "pointmass-reach"
    config = WandbConfig()
    task_config = PointReachConfig(
        observation_type="visual_observations",
        reward_type="dense_biased_negative_distance_reward",
        goal_distance_threshold=0.05,
        max_step_size=0.1,
        max_control_steps_per_episode=15,
        image_resolution=64,
    )

dmc_env = Environment(PointMassReachTask(task_config), strip_singleton_obs_buffer_dim=True)

sb3_sac(dmc_env, config, task_config, log_dir, tags=["pointmass_reach"])
