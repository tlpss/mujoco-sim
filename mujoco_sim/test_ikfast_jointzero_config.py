import numpy as np

from ur_ikfast.ur_ikfast.ur_kinematics import URKinematics

"""Real controller zero-joint pose is in -x,-y

UR IKFAST (& URDFs) in +x,+y??
"""
if __name__ == "__main__":
    zero_joint_pose = URKinematics("ur3").forward(np.zeros(6))
    print(zero_joint_pose)
