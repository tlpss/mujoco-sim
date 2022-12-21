import abc
import dataclasses

from dm_control import composer


@dataclasses.dataclass
class TaskConfig:
    physics_timestep: float = 0.002  # MJC default
    control_timestep: float = 0.06  # 30 physics steps
    max_control_steps_per_episode: int = 50


class RobotTask(composer.Task, abc.ABC):
    """Custom base class for Robot Tasks."""

    CONFIG_CLASS = TaskConfig

    def __init__(self, config: CONFIG_CLASS = None) -> None:
        super().__init__()
        self.config: RobotTask.CONFIG_CLASS = self.CONFIG_CLASS() if config is None else config

        self._task_observables = None  # Dict(str, Observable)
        # TODO: Robot, EEF and arena could also be created here

    def initialize_episode(self, physics, random_state):
        # use counter instead of physics.time() to deal with
        # synchronous interactions (varying #physics.steps() per control step)
        self.episode_step = 0

    def before_step(self, physics, action, random_state):
        self.episode_step += 1

    def _configure_observables(self):
        """
        enables the appropriate observables and optionally
        configures them with aggregators, noise...
        """
        raise NotImplementedError

    def is_task_accomplished(self) -> bool:
        """
        returns true if the task is solved
        """
        raise NotImplementedError

    def should_terminate_episode(self, physics) -> bool:
        accomplished = self.is_task_accomplished(physics)
        time_limit = self.episode_step >= self.config.max_control_steps_per_episode

        return accomplished or time_limit

    def get_discount(self, physics):
        # properly handle truncation of inifite horizon tasks
        # by returning 0.0 only if the task is accomplished
        # cf gym v0.26 breaking change
        return 0.0 if self.is_task_accomplished(physics) else 1.0

    @property
    def task_observables(self):
        return {name: obs for (name, obs) in self._task_observables.items() if obs.enabled}
