import numpy as np
from dm_control import mjcf

from mujoco_sim.entities.robots.robot import UR5e


def test_sim_ur_frame_matches_real():
    """check if the base frame of the sim robot matches the base frame of a real robot, by checking if a joint config results in the same EEF pose."""

    robot = UR5e()
    physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)

    joint_config = np.zeros(6)

    physics.bind(robot.mjcf_model.find_all("joint")).qpos = joint_config

    # get attachment site pose
    attachment_site = robot.mjcf_model.find("site", "attachment_site")
    attachment_site_pose = physics.bind(attachment_site).xpos
    attachment_site_orientation = physics.bind(attachment_site).xmat.reshape(3, 3)
    pose = np.eye(4)
    pose[:3, :3] = attachment_site_orientation
    pose[:3, 3] = attachment_site_pose

    import ur_analytic_ik

    FK_pose = ur_analytic_ik.ur5e.forward_kinematics(*joint_config)

    assert np.allclose(pose, FK_pose, atol=1e-2), f"Pose mismatch: {pose} vs {FK_pose}"
