import dataclasses


@dataclasses.dataclass
class TaskConfig:
    physics_timestep: float = 0.002  # MJC default
    control_timestep: float = 0.06  # 30 physics steps
    max_control_steps_per_episode: int = 50
