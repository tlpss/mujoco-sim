import time

import numpy as np

from ur_ikfast.ur_ikfast.ur_kinematics import URKinematics

"""Real controller zero-joint pose is in -x,-y

UR IKFAST (& URDFs) in +x,+y??
"""


def profile_forward():
    ikfast = URKinematics("ur5e")
    start = time.time()
    for i in range(1000):
        res = ikfast.forward(np.random.uniform(-np.pi, np.pi, 6))
        assert res.any() is not None
    stop = time.time()
    return (stop - start) / 1000


def profile_inverse():
    ikfast = URKinematics("ur5e")
    start = time.time()
    for i in range(1000):
        quat = np.random.uniform(0, 1, 4)
        quat /= np.linalg.norm(quat)
        ikfast.inverse(np.concatenate([np.random.uniform(-0.7, 0.7, 3), quat]))
    stop = time.time()
    return (stop - start) / 1000


if __name__ == "__main__":
    zero_joint_pose = URKinematics("ur3").forward(np.zeros(6))
    print(zero_joint_pose)
    print(profile_forward())
    print(profile_inverse())
    import cProfile

    cProfile.run("profile_inverse()")
