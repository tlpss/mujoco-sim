import enum
import time
from pathlib import Path

import gymnasium
import torch


class DatasetRecorder:
    def __init__(self):
        pass

    def start_episode(self):
        raise NotImplementedError

    def record(self, obs, action, reward, done, info):
        raise NotImplementedError

    def save_episode(self):
        raise NotImplementedError

    def finish_recording(self):
        pass


class DummyDatasetRecorder(DatasetRecorder):
    def start_episode(self):
        print("starting dataset episode recording")

    def record(self, obs, action, reward, done, info):
        print("recording step")

    def save_episode(self):
        print("saving dataset episode")


class LeRobotDatasetRecorder(DatasetRecorder):
    DEFAULT_FEATURES = {
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "seed": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
    }

    def __init__(self, env: gymnasium.Env, root_dataset_dir: Path, dataset_name: str, fps: int):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.root_dataset_dir = root_dataset_dir
        self.dataset_name = dataset_name
        self.fps = fps

        self.n_recorded_episodes = 0
        self.key_mapping_dict = {}

        features = self.DEFAULT_FEATURES.copy()
        # add images to features

        # uses the lerobot convention to map to 'observation.image' keys
        # and stores as video.

        assert isinstance(env.observation_space, gymnasium.spaces.Dict), "Observation space should be a dict"
        self.image_keys = [key for key in env.observation_space.spaces.keys() if "image" in key]
        num_cameras = len(self.image_keys)
        for key in self.image_keys:
            shape = env.observation_space.spaces[key].shape
            if not key.startswith("observation.image"):
                lerobot_key = f"observation.image.{key}"
                self.key_mapping_dict[key] = lerobot_key
                features[lerobot_key] = {"dtype": "video", "names": ["channel", "height", "width"], "shape": shape}

        # state observations
        self.state_keys = [key for key in env.observation_space.spaces.keys() if key not in self.image_keys]
        for key in self.state_keys:
            shape = env.observation_space.spaces[key].shape
            features[key] = {"dtype": "float32", "shape": shape, "names": None}

        # add action to features
        features["action"] = {"dtype": "float32", "shape": env.action_space.shape, "names": None}

        print(f"Features: {features}")
        # create the dataset
        self.lerobot_dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=self.fps,
            root=self.root_dataset_dir,
            features=features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * num_cameras,
        )

    def start_episode(self):
        pass

    def record(self, obs, action, reward, done, info):
        env_timestamp = info.get("timestamp", self.lerobot_dataset.episode_buffer["size"] / self.fps)

        frame = {
            "action": torch.from_numpy(action),
            "next.reward": torch.tensor(reward),
            "next.success": torch.tensor(done),
            "seed": torch.tensor(0),  # TODO: store the seed
            "timestamp": env_timestamp,
        }
        for key in self.image_keys:
            lerobot_key = self.key_mapping_dict.get(key, key)
            frame[lerobot_key] = obs[key]

        for key in self.state_keys:
            frame[key] = torch.from_numpy(obs[key])

        self.lerobot_dataset.add_frame(frame)

    def save_episode(self):
        self.lerobot_dataset.save_episode(task="")
        self.n_recorded_episodes += 1

    def finish_recording(self):
        # computing statistics
        self.lerobot_dataset.consolidate()


class UI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Demonstration Collection")

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "recording"
                elif event.key == pygame.K_d:
                    return "discard"
                elif event.key == pygame.K_f:
                    return "finish"
                elif event.key == pygame.K_q:
                    return "quit"
        return None

    def update_display_color(self, color):
        self.screen.fill(color)
        pygame.display.flip()

    # def add_render_to_screen(self, img: np.ndarray):
    #     assert img.shape[2] == 3, "Image should be in RGB format"
    #     img = img * 255

    #     surf = pygame.surfarray.make_surface(img)
    #     self.screen.blit(surf, (50,50))
    #     pygame.display.update()


class StateMachine:
    STATES = enum.Enum("states", "waiting start recording discard finish quit")

    def __init__(self):
        self.state = self.STATES.waiting
        self.ui = UI()

    def update_state_from_keyboard(self):
        event = self.ui.process_events()
        if event:
            self.state = self.STATES[event]
        # print(f"State: {self.state}")

        if self.state is self.STATES.recording:
            self.ui.update_display_color((0, 0, 255))
        else:
            self.ui.update_display_color((0, 0, 0))

        # print state to UI
        font = pygame.font.Font(None, 36)
        text = font.render(f"State: {self.state.name}", True, (255, 255, 255))
        self.ui.screen.blit(text, (0, 0))
        pygame.display.update()


def collect_demonstrations(agent_callable, env, dataset_recorder):
    # create opencv screen for rendering
    import cv2

    # window = cv2.namedWindow("Demonstration Collection Rendering", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Demonstration Collection Rendering", 640, 480)
    # start the UI
    state_machine = StateMachine()
    while state_machine.state is not state_machine.STATES.quit:
        while state_machine.state is state_machine.STATES.waiting:
            time.sleep(1)
            state_machine.update_state_from_keyboard()

        if state_machine.state is state_machine.STATES.quit:
            break

        env.reset()
        dataset_recorder.start_episode()
        done = False

        while not done and state_machine.state is state_machine.STATES.recording:
            action = agent_callable(env)
            obs, reward, done, info = env.step(action)
            dataset_recorder.record(obs, action, reward, done, info)
            img = env.render()
            # reshape to (640,480)
            img = cv2.resize(img, (640, 480))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Demonstration Collection Rendering", img)
            # cv2.waitKey(1)
            # add img to the screen
            state_machine.update_state_from_keyboard()

        print("Episode is done, decide what to do next")
        while done and state_machine.state is state_machine.STATES.recording:
            state_machine.update_state_from_keyboard()

        if state_machine.state is state_machine.STATES.discard:
            print("not saving the episode")

        if state_machine.state is state_machine.STATES.finish:
            dataset_recorder.save_episode()

        state_machine.state = state_machine.STATES.waiting

    dataset_recorder.finish_recording()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import pygame
    from dm_control.composer import Environment as DMEnvironment

    from mujoco_sim.environments.dmc2gym import DMCWrapper
    from mujoco_sim.environments.tasks.point_reach import (
        VISUAL_OBS,
        PointMassReachTask,
        PointReachConfig,
        create_demonstation_policy,
    )

    task = PointMassReachTask(PointReachConfig(observation_type=VISUAL_OBS))
    dmc_env = DMEnvironment(task, strip_singleton_obs_buffer_dim=True)
    env = DMCWrapper(dmc_env)

    action_callable = create_demonstation_policy(dmc_env)
    import datetime

    id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    dataset_recorder = LeRobotDatasetRecorder(
        env, Path(__file__).parent / "dataset" / f"{id}", "point_reach", round(1 / env._env.control_timestep())
    )
    collect_demonstrations(action_callable, env, dataset_recorder)
