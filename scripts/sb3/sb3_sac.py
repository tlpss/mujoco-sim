import dataclasses
from pathlib import Path

import torch
import wandb
from dm_control.composer import Environment
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.sac.sac import SAC
from wandb.integration.sb3 import WandbCallback

from mujoco_sim.environments.dmc2gym import DMCEnvironmentAdapter
from mujoco_sim.gym_video_wrapper import VideoRecorderWrapper


def sb3_sac(dmc_env: Environment, hparam_config, task_config, log_dir: Path, tags: list = None):
    config_dict = dataclasses.asdict(hparam_config)
    config_dict.update(dataclasses.asdict(task_config))
    print(f"{config_dict=}")
    run = wandb.init(project="mujoco-sim", config=config_dict, sync_tensorboard=True, mode="online", tags=tags)
    wandb.config.update(config_dict)  # strange but wandb did not set dict in init..?

    torch.manual_seed(hparam_config.seed)
    gym_env = DMCEnvironmentAdapter(dmc_env, flatten_observation_space=False, render_camera_id=0)
    gym_env = VideoRecorderWrapper(
        gym_env, log_dir / f"{run.name}_videos", capture_every_n_episodes=20, log_wandb=True, rescale_video_factor=1
    )
    print(gym_env.action_space)
    print(gym_env.observation_space)

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
        learning_rate=hparam_config.lr,
        tau=hparam_config.tau,
        gamma=hparam_config.gamma,
        seed=hparam_config.seed,
        ent_coef=hparam_config.entropy_coefficient,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        buffer_size=200_000,
        device="cuda",
        train_freq=(2, "episode"),
        gradient_steps=hparam_config.gradient_steps,
        batch_size=hparam_config.batch_size,
    )

    wandb_callback = WandbCallback(1, log_dir / "models", 0, 0, "all")
    eval_callback = EvalCallback(gym_env, n_eval_episodes=5, eval_freq=10000)

    print(sac.policy.features_extractor)
    sac.learn(hparam_config.timesteps, progress_bar=True, callback=[wandb_callback, eval_callback])
