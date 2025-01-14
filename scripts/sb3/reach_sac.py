import copy
import dataclasses

import torch
import wandb
from dm_control.composer import Environment
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.sac.sac import SAC
from wandb.integration.sb3 import WandbCallback

from mujoco_sim import _LOGGING_DIR
from mujoco_sim.environments.dmc2gym import DMCEnvironmentAdapter
from mujoco_sim.environments.tasks.robot_reach import RobotReachConfig, RobotReachTask
from mujoco_sim.gym_video_wrapper import VideoRecorderWrapper


@dataclasses.dataclass
class HyperConfig:
    lr: float = 5e-4
    tau: float = 0.005
    gamma: float = 0.99
    timesteps: int = 200000
    seed: int = 2022
    entropy_coefficient: int = 0.005
    batch_size: int = 32
    gradient_steps: int = 1
    num_envs: int = 1


if __name__ == "__main__":

    log_dir = _LOGGING_DIR / "robot-reach"
    config = HyperConfig()
    task_config = RobotReachConfig(
        observation_type=RobotReachConfig.VISUAL_OBS,
        max_control_steps_per_episode=30,
    )

    config_dict = dataclasses.asdict(config)
    config_dict.update(dataclasses.asdict(task_config))
    print(f"{config_dict=}")
    run = wandb.init(
        project="mujoco-sim", config=config_dict, sync_tensorboard=True, mode="online", tags=["planar_reach"]
    )
    wandb.config.update(config_dict)  # strange but wandb did not set dict in init..?
    config = wandb.config  # get possibly updated config
    print(config)
    torch.manual_seed(config.seed)
    print("executing")

    def create_env(rank, seed, task_config):
        """
        if you do not create an inner function
        and just return the env using a lambda in the SubProcVecEnv init,
        the arguments seem to be copied by reference or something.
        So they would all have the same rank (of the last env..)

        This function follows the pattern at
        https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/3_multiprocessing.ipynb
        Args:
            rank (_type_): _description_
            seed (_type_): _description_
            task_config (_type_): _description_
        """

        def _create():
            print("creating env")
            dmc_env = Environment(RobotReachTask(task_config), strip_singleton_obs_buffer_dim=True)
            env = DMCEnvironmentAdapter(dmc_env, flatten_observation_space=False, render_camera_id=0)
            print(rank)
            if rank == 0:
                print("wrapping rank 0 env")
                env = VideoRecorderWrapper(
                    env,
                    log_dir / f"{run.name}_videos",
                    capture_every_n_episodes=20,
                    log_wandb=True,
                    rescale_video_factor=2,
                )
                # does not seem to work anymore?
                env = Monitor(env)
            env.seed(seed + rank)
            return env

        return _create

    # make deepcopy of seed to avoid
    # a recursion error on the wandb.config.get()
    # in the subproc create env
    seed = copy.deepcopy(config.seed)
    if config.num_envs == 1:
        vecenv = DummyVecEnv([create_env(0, seed, task_config)])
    else:
        vecenv = SubprocVecEnv([create_env(i, seed, task_config) for i in range(config.num_envs)])

    print(vecenv.action_space)
    print(vecenv.observation_space)

    # Multi-input policy: CNN for images & then concat with non-image
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    # #multiple-inputs-and-dictionary-observations
    policy_kwargs = {
        "net_arch": dict(qf=[64, 64], pi=[64, 64]),  # extra NN in the CNN output! so conv2d-conv2d-conv2d-NN-NN
        "share_features_extractor": True,
    }
    sac = SAC(
        "MultiInputPolicy",
        env=vecenv,
        verbose=1,
        learning_rate=config.lr,
        tau=config.tau,
        gamma=config.gamma,
        seed=config.seed,
        ent_coef=config.entropy_coefficient,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        buffer_size=200_000,
        device="cpu",
        # train_freq=(2,"episode"),
        # gradient_steps=config.gradient_steps,
        batch_size=config.batch_size,
    )

    wandb_callback = WandbCallback(1, log_dir / "models", 0, 0, "all")

    eval_callback = EvalCallback(vecenv, n_eval_episodes=5, eval_freq=10000)

    print(sac.policy.features_extractor)
    sac.learn(config.timesteps, progress_bar=True, callback=[wandb_callback, eval_callback])
    ## profiling
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # sb3_sac(dmc_env, config, task_config, log_dir, tags=["planar_push"])
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.dump_stats('sac_sb3_robot_push.prof')
    # stats.print_stats(50)
