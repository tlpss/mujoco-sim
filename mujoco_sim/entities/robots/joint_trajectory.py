from __future__ import annotations

import dataclasses

from mujoco_sim.type_aliases import JOINT_CONFIGURATION_TYPE


@dataclasses.dataclass
class Waypoint:
    joint_positions: JOINT_CONFIGURATION_TYPE
    timestep: float


class JointTrajectory:
    """A container to hold a joint trajectory defined by a number of (key) waypoints.
    The trajectory is defined by linear interpolation between the waypoints.

    Note that this might not be a smooth trajectory, as the joint velocities/accelerations are not guarantueed to be continuous
    for the interpolated trajectory. To overcome this, use an appropriate trajectory generator and timestep to provide the
    waypoints in this trajectory so that the discontinuities that are introduced by the linear interpolation are small.
    """

    def __init__(self, waypoints: list[Waypoint]):
        # # sort by timestep to make sure that the waypoints are in order
        self.waypoints = sorted(waypoints, key=lambda x: x.timestep)

    def get_nearest_waypoints(self, t: float) -> tuple[Waypoint, Waypoint]:
        for i in range(len(self.waypoints) - 1):
            if t >= self.waypoints[i].timestep and t <= self.waypoints[i + 1].timestep:
                return self.waypoints[i], self.waypoints[i + 1]
        raise ValueError("should not be here")

    def _clip_timestep(self, t: float) -> float:
        """clips the timestep to the range of the waypoints"""

        # for scalars, min(max()) is about 100x faster than np.clip()
        # https://github.com/numpy/numpy/issues/14281#issuecomment-552472647

        return min(max(t, self.waypoints[0].timestep), self.waypoints[-1].timestep)

    def get_target_joint_positions(self, t: float) -> JOINT_CONFIGURATION_TYPE:
        """returns the target joint positions at time t by linear interpolation between the waypoints"""
        t = self._clip_timestep(t)
        previous_waypoint, next_waypoint = self.get_nearest_waypoints(t)
        t0, q0 = previous_waypoint.timestep, previous_waypoint.joint_positions
        t1, q1 = next_waypoint.timestep, next_waypoint.joint_positions
        return q0 + (q1 - q0) * (t - t0) / (t1 - t0)

    def get_target_joint_velocities(self, t: float) -> JOINT_CONFIGURATION_TYPE:
        """returns the target joint velocities at time t by linear interpolation between the waypoints"""
        t = self._clip_timestep(t)
        previous_waypoint, next_waypoint = self.get_nearest_waypoints(t)
        t0, q0 = previous_waypoint.timestep, previous_waypoint.joint_positions
        t1, q1 = next_waypoint.timestep, next_waypoint.joint_positions
        return (q1 - q0) / (t1 - t0)

    def is_finished(self, t: float) -> bool:
        """returns True if the trajectory is finished at time t"""
        return t >= self.waypoints[-1].timestep
