import enum
import time
from pathlib import Path

import gymnasium
import pygame
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

    def __init__(self, env: gymnasium.Env, root_dataset_dir: Path, dataset_name: str, fps: int, use_videos=True):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.root_dataset_dir = root_dataset_dir
        self.dataset_name = dataset_name
        self.fps = fps

        self._n_recorded_episodes = 0
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

            if not key.startswith("observation.images"):
                lerobot_key = f"observation.images.{key}"
                self.key_mapping_dict[key] = lerobot_key

            lerobot_key = self.key_mapping_dict.get(key, key)
            if "/" in lerobot_key:
                self.key_mapping_dict[key] = lerobot_key.replace("/", "_")
            lerobot_key = self.key_mapping_dict[key]
            if use_videos:
                features[lerobot_key] = {"dtype": "video", "names": ["channel", "height", "width"], "shape": shape}
            else:
                features[lerobot_key] = {"dtype": "image", "shape": shape, "names": None}

        # state observations
        self.state_keys = [key for key in env.observation_space.spaces.keys() if key not in self.image_keys]
        for key in self.state_keys:
            shape = env.observation_space.spaces[key].shape
            features[key] = {"dtype": "float32", "shape": shape, "names": None}

        # add single 'state' observation that concatenates all state observations
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (sum([env.observation_space.spaces[key].shape[0] for key in self.state_keys]),),
            "names": None,
        }
        # add action to features
        features["action"] = {"dtype": "float32", "shape": env.action_space.shape, "names": None}

        print(f"Features: {features}")
        # create the dataset
        self.lerobot_dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=self.fps,
            root=self.root_dataset_dir,
            features=features,
            use_videos=use_videos,
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

        # concatenate all 'state' observations into a single tensor
        state = torch.cat([frame[key].flatten() for key in self.state_keys])
        frame["observation.state"] = state

        self.lerobot_dataset.add_frame(frame)

    def save_episode(self):
        self.lerobot_dataset.save_episode(task="")
        self._n_recorded_episodes += 1

    def finish_recording(self):
        # computing statistics
        self.lerobot_dataset.consolidate()

    @property
    def n_recorded_episodes(self):
        return self._n_recorded_episodes


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

    def update_UI(self, state, n_demos_collected, info):
        # make screen all black
        self.screen.fill((0, 0, 0))

        # add text to the screen
        font = pygame.font.Font(None, 36)
        text = font.render(f"State: {state}", True, (255, 255, 255))
        self.screen.blit(text, (50, 50))

        # add # collected demos
        text = font.render(f"Collected Demos: {n_demos_collected}", True, (255, 255, 255))
        self.screen.blit(text, (50, 100))

        # add info
        text = font.render(f"Info: {info}", True, (255, 255, 255))
        self.screen.blit(text, (50, 150))

        pygame.display.flip()

    # def add_render_to_screen(self, img: np.ndarray):
    #     assert img.shape[2] == 3, "Image should be in RGB format"
    #     img = img * 255

    #     surf = pygame.surfarray.make_surface(img)
    #     self.screen.blit(surf, (50,50))
    #     pygame.display.update()


class StateMachine:
    STATES = enum.Enum("states", "waiting recording discard finish quit")

    def __init__(self):
        self.state = self.STATES.waiting
        self.ui = UI()

    def update_state_from_keyboard(self):
        event = self.ui.process_events()
        if event:
            if event == self.STATES.discard.name or event == self.STATES.finish.name:
                if self.state == self.STATES.recording:
                    self.state = self.STATES[event]
            if event == self.STATES.recording.name:
                if self.state == self.STATES.waiting:
                    self.state = self.STATES[event]
                else:
                    print("Cannot start recording from this state")

            if event == self.STATES.quit.name:
                self.state = self.STATES[event]


def collect_demonstrations(agent_callable, env, dataset_recorder):
    # create opencv screen for rendering
    import cv2

    # window = cv2.namedWindow("Demonstration Collection Rendering", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Demonstration Collection Rendering", 640, 480)
    # start the UI
    state_machine = StateMachine()
    while state_machine.state is not state_machine.STATES.quit:
        while state_machine.state is state_machine.STATES.waiting:
            time.sleep(0.1)
            state_machine.update_state_from_keyboard()
            state_machine.ui.update_UI(state_machine.state, dataset_recorder.n_recorded_episodes, "")

        if state_machine.state is state_machine.STATES.quit:
            break

        obs, info = env.reset()
        dataset_recorder.start_episode()
        done = False

        while not done and state_machine.state is state_machine.STATES.recording:
            action = agent_callable(env)
            next_obs, reward, termination, truncation, info = env.step(action)
            done = termination or truncation
            dataset_recorder.record(obs, action, reward, done, info)
            obs = next_obs

            img = env.render()
            # reshape to (640,480)
            img = cv2.resize(img, (640, 480))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Demonstration Collection Rendering", img)
            # cv2.waitKey(1)
            # add img to the screen
            input("Press enter to continue")

            state_machine.update_state_from_keyboard()
            state_machine.ui.update_UI(state_machine.state, dataset_recorder.n_recorded_episodes, "")

        print("Episode is done, decide what to do next")
        while done and state_machine.state is state_machine.STATES.recording:
            state_machine.update_state_from_keyboard()
            state_machine.ui.update_UI(state_machine.state, dataset_recorder.n_recorded_episodes, "save episode?")

            # hardcode
            state_machine.state = state_machine.STATES.finish

        if state_machine.state is state_machine.STATES.discard:
            print("not saving the episode")

        if state_machine.state is state_machine.STATES.finish:
            dataset_recorder.save_episode()

        state_machine.state = state_machine.STATES.waiting

    dataset_recorder.finish_recording()
    cv2.destroyAllWindows()


# data collection that is not blocking on user input or uses a UI, to use with remote machines for sim envs


def collect_demonstrations_non_blocking(agent_callable, env, dataset_recorder, n_episodes=50):
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        dataset_recorder.start_episode()
        while not done:
            action = agent_callable(env)
            new_obs, reward, termination, truncation, info = env.step(action)
            done = termination or truncation
            dataset_recorder.record(obs, action, reward, done, info)
            obs = new_obs
        dataset_recorder.save_episode()

    dataset_recorder.finish_recording()


if __name__ == "__main__":
    # import pygame
    import mujoco_sim  # noqa

    env = gymnasium.make("mujoco_sim/robot_push_button_visual-v0")

    dmc_env = env.unwrapped.dmc_env

    action_callable = dmc_env.task.create_demonstration_policy(dmc_env)
    import datetime

    id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    dataset_recorder = LeRobotDatasetRecorder(
        env,
        Path(__file__).parent / "dataset" / f"{id}",
        "point_reach",
        round(1 / env.unwrapped.dmc_env.control_timestep()),
        use_videos=False,
    )
    collect_demonstrations_non_blocking(action_callable, env, dataset_recorder, n_episodes=100)
    # set MUJOCO_GL=egl to run this on a remote machine
    # collect_demonstrations_non_blocking(action_callable, env, dataset_recorder, n_episodes=300)
