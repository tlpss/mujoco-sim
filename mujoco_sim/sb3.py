import dataclasses

import numpy as np
import torch
import wandb
from dm_control import viewer
from dm_control.composer import Environment
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.sac.sac import SAC
from wandb.integration.sb3 import WandbCallback

from mujoco_sim import _LOGGING_DIR
from mujoco_sim.environments.dmc2gym import DMCWrapper
from mujoco_sim.environments.tasks.point_reach import PointMassReachTask, PointReachConfig
from mujoco_sim.gym_video_wrapper import VideoRecorderWrapper

if __name__ == "__main__":

    @dataclasses.dataclass
    class WandbConfig:
        lr = 3e-4
        tau = 0.005
        gamma = 0.95
        timesteps = 1_000_000
        seed = 2022
        entropy_coefficient = 0.001

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

    config_dict = dataclasses.asdict(config)
    config_dict.update(dataclasses.asdict(task_config))
    print(f"{config_dict=}")
    run = wandb.init(
        project="mujoco-sim", config=config_dict, sync_tensorboard=True, mode="online", tags=["pointmass_reach"]
    )
    wandb.config.update(config_dict)  # strange but wandb did not set dict in init..?

    torch.manual_seed(config.seed)
    dmc_env = Environment(PointMassReachTask(task_config), strip_singleton_obs_buffer_dim=True)
    gym_env = DMCWrapper(dmc_env, flatten_observation_space=False, render_camera_id=1)
    gym_env = VideoRecorderWrapper(
        gym_env, log_dir / f"{run.name}_videos", capture_every_n_episodes=50, log_wandb=True, rescale_video_factor=1
    )
    print(gym_env.action_space)
    print(gym_env.observation_space)

    obs = gym_env.reset()

    # Multi-input policy: CNN for images & then concat with non-image
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    # #multiple-inputs-and-dictionary-observations
    policy_kwargs = {
        "net_arch": dict(qf=[64], pi=[32]),  # extra NN in the CNN output! so conv2d-conv2d-conv2d-NN-NN
        "share_features_extractor": True,
    }
    sac = SAC(
        "MultiInputPolicy",
        env=gym_env,
        verbose=1,
        learning_rate=config.lr,
        tau=config.tau,
        gamma=config.gamma,
        seed=config.seed,
        ent_coef=config.entropy_coefficient,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        buffer_size=200_000,
        device="cuda",
        # train_freq=(2,"episode"),
        gradient_steps=-1,
        batch_size=64,
    )

    wandb_callback = WandbCallback(1, log_dir / "models", 0, 0, "all")
    eval_callback = EvalCallback(gym_env, n_eval_episodes=5, eval_freq=5000)

    print(sac.policy.features_extractor)
    sac.learn(config.timesteps, progress_bar=True, callback=[wandb_callback, eval_callback])

    def policy(time_step):
        action = sac.predict(gym_env.env._get_obs(time_step), deterministic=False)[0]
        action = np.array(action)
        return action

    viewer.launch(dmc_env, policy)
