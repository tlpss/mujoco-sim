from dataclasses import dataclass, field
import multiprocessing
from pathlib import Path
from lerobot.common.envs.configs import EnvConfig
from lerobot.common.utils.utils import init_logging
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.scripts.train import train
import mujoco_sim
from mujoco_sim.demonstration_collection import collect_demonstrations_non_blocking, LeRobotDatasetRecorder
import gymnasium


import os 
import shutil
import datetime

from mujoco_sim.entities.camera import camera_orientation_from_look_at
# if os.path.exists("tmp/dataset/robot_push_button"):
#     shutil.rmtree("tmp/dataset/robot_push_button")

# if os.path.exists("tmp/outputs/train"):
#     shutil.rmtree("tmp/outputs/train")

 ## training
@EnvConfig.register_subclass("mujoco_sim")
@dataclass
class PushButtonConfig(EnvConfig):
    task: str = "robot_push_button_visual-v0"
    fps: int = 10
    episode_length: int = 100
    image_resolution: int = 128
    camera_position: list[float] = field(default_factory=lambda: [0, 0, 1.0])
    camera_orientation: list[float] = field(default_factory=lambda: [0, 0, 0, 1])
    use_wrist_camera: bool = False
    
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": "action",
            "agent_pos": "observation.state",
            "pixels/scene": f"observation.images.scene",
            "pixels/wrist": f"observation.images.wrist",
        }
    )

    def __post_init__(self):
        self.features["pixels/scene"] = PolicyFeature(type=FeatureType.VISUAL, shape=(self.image_resolution, self.image_resolution, 3))
        if self.use_wrist_camera:
            self.features["pixels/wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=(self.image_resolution, self.image_resolution, 3))
    
    
    
    @property
    def gym_kwargs(self) -> dict:
        return {
            "scene_camera_position": self.camera_position,
            "scene_camera_orientation": self.camera_orientation,
            "use_wrist_camera": False,
            "image_resolution": self.image_resolution,
        }

def run(seed,camera_position, camera_orientation):
    os.makedirs("tmp/dataset/robot_push_button", exist_ok=True)
    dataset_path = f"tmp/dataset/robot_push_button/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    ## create dataset and store
    env_config = PushButtonConfig(
            camera_position=camera_position,
            camera_orientation=camera_orientation,
            use_wrist_camera=False,
            image_resolution=256,
        )
    env = gymnasium.make("mujoco_sim/robot_push_button_visual-v0", **env_config.gym_kwargs)
    env.seed(seed)
    env.reset()
    dataset_recorder = LeRobotDatasetRecorder(env,dataset_path,f"tlips/{camera_position[0]:.2f}_{camera_position[1]:.2f}_{camera_position[2]:.2f}",10,True)

    dmc_env = env.unwrapped.dmc_env
    action_callable = dmc_env.task.create_demonstration_policy(dmc_env)

    collect_demonstrations_non_blocking(action_callable, env, dataset_recorder, n_episodes=150)


   


    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="whatever/whatever", root=dataset_path),
        policy=ACTConfig(
            device='cuda',
            n_obs_steps=1,
            chunk_size=20,
            n_action_steps=10,
            input_features={
                "observation.images.scene": PolicyFeature(type=FeatureType.VISUAL, shape=(env_config.image_resolution, env_config.image_resolution, 3)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            },
            dim_feedforward=1024,
            dim_model=512,
            optimizer_lr=2e-5,

            ),

        eval=EvalConfig(n_episodes=20, batch_size=10),
        env=env_config,
        wandb=WandBConfig(enable=True, disable_artifact=True,project="camera-placement"),
        output_dir=Path(f"tmp/outputs/train/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"), 
        job_name=f"robot-push-button-[{camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}]",
        resume=False,
        seed=seed,
        num_workers=4,
        batch_size=64,
        steps=100_000,
        eval_freq=10_000,
        log_freq=1000,
        save_freq=100_000
    )

    # train policy

    print(config)
    train(config)


# load the latest checkpoint and eval further?


if __name__ == "__main__":
    import numpy as np 
    look_at_position = np.array([0.0, -0.4, 0])


    camera_positions = [
        # np.array([0.0, -1., 0.1]),
        # np.array([0, -1., 1]),
        np.array([0.0, -0.41, 0.9]),
        np.array([1,-1,1]),
        np.array([0,-1., .4]),
        np.array([0.5,-0.8, .4]),
        np.array([0,-1.5,0.5])
    ]

    camera_orientations = [
        camera_orientation_from_look_at(camera_position,look_at_position)
        for camera_position in camera_positions
    ]

    # create new wandb run
    import wandb 
    # check if run is active, if so, finish it
    for seed in [2025]: #,2026,2027]:
        for camera_position, camera_orientation in zip(camera_positions, camera_orientations):
            if wandb.run is not None:
                print("Wandb run is active, finishing it...")
                wandb.finish()
            print(f"Running with camera position: {camera_position} and camera orientation: {camera_orientation} and seed: {seed}")
            init_logging()

            run(seed,camera_position.tolist(), camera_orientation.tolist())
