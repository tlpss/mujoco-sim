import numpy as np
from dm_control import mjcf

from mujoco_sim.entities.robots.robot import UR5e


def test_moveJ():
    robot = UR5e()
    physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)

    # force robot to move to a specific pose
    joints = np.array([0.0, -0.5, 0.5, -0.5, -0.5, -0.5]) * np.pi
    # robot.set_joint_positions(physics, joints)

    # target_pose  = np.array([0.1, 0.3, 0.5, 1,0,0,0])

    print(f"initial joint positions: {robot.get_joint_positions(physics)}")
    robot.moveJ(physics, joints, 1.0)
    for _ in range(10000):
        robot.before_substep(physics, None)
        physics.step()

    print(f"final joint positions: {robot.get_joint_positions(physics)}")
    print(f"target joint positions: {joints}")

    assert np.allclose(
        robot.get_joint_positions(physics), joints, atol=1e-2
    ), f"Joint mismatch: {robot.get_joint_positions(physics)} vs {joints}"


def test_moveJ_IK():
    robot = UR5e()
    physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)

    # set physics step size
    print(f"physics timestep: {physics.timestep()}")

    # force robot to move to a specific pose
    joints = np.array([0.0, -0.5, 0.5, -0.5, -0.5, -0.5]) * np.pi
    robot.set_joint_positions(physics, joints)

    target_pose = np.array([0.1, 0.3, 0.5, 1, 0, 0, 0])

    print(f"initial pose: {robot.get_tcp_pose(physics)}")
    robot.movej_IK(physics, target_pose, 1.0)
    for _ in range(6000):
        robot.before_substep(physics, None)
        physics.step()

    print(f"final pose: {robot.get_tcp_pose(physics)}")
    print(f"target pose: {target_pose}")
    assert np.allclose(
        robot.get_tcp_pose(physics), target_pose, atol=1e-2
    ), f"Pose mismatch: {robot.get_tcp_pose(physics)} vs {target_pose}"


def test_servoL():
    robot = UR5e()
    physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)

    # set physics step size
    print(f"physics timestep: {physics.timestep()}")

    # force robot to move to a specific pose
    joints = np.array([0.0, -0.5, 0.5, -0.5, -0.5, -0.5]) * np.pi
    robot.set_joint_positions(physics, joints)

    target_pose = np.array([0.1, 0.3, 0.5, 1, 0, 0, 0])

    print(f"initial pose: {robot.get_tcp_pose(physics)}")
    for _ in range(20):
        robot.servoL(physics, target_pose, 0.2)
        for _ in range(100):
            robot.before_substep(physics, None)
            physics.step()

    print(f"final pose: {robot.get_tcp_pose(physics)}")
    print(f"target pose: {target_pose}")

    assert np.allclose(
        robot.get_tcp_pose(physics), target_pose, atol=1e-2
    ), f"Pose mismatch: {robot.get_tcp_pose(physics)} vs {target_pose}"
