import dataclasses

from dm_control import composer


@dataclasses.dataclass
class TaskConfig:
    physics_timestep: float = 0.002  # MJC default
    control_timestep: float = 0.06  # 30 physics steps
    max_control_steps_per_episode: int = 50


class RobotTaskConfig(TaskConfig):
    robot = None
    gripper = None
    arena = None


class RobotTask(composer.Task):
    """Custom base class for Robot Tasks."""

    CONFIG_CLASS = None

    def __init__(self, config: CONFIG_CLASS = None) -> None:
        super().__init__()
        self.config = self.CONFIG_CLASS() if config is None else config

    def _configure_observables(self):
        """
        enables the appropriate observables and optionally
        configures them with aggregators, noise...
        """
