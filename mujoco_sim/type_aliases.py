import numpy as np

VECTOR_TYPE = np.ndarray
""" a 3D vector (x,y,z) that represents position, velocity,..."""

ORIENTATION_TYPE = np.ndarray
""" a 4D vector containing a scalar-last quaternion (x,y,z,w)[radians] that represents the orientation of a frame in another frame"""

POSE_TYPE = np.ndarray
""" a 7D vector containing and SE3 pose as  (position [m],orientation as scalar-last quaternion [radians])"""

JOINT_CONFIGURATION_TYPE = np.ndarray
"""" an n-D vector containing joint positions/velocities/accelerations [radians]"""

WRENCH_TYPE = np.ndarray
""" a 6D vector containing a 3D force and 3D torque"""

LINEAR_VELOCITY_TYPE = np.ndarray
""" a 3D vector containing a linear (x,y,z) velocity [m/s] or acceleration"""

ANGULAR_VELOCITY_TYPE = np.ndarray
""" A 3D vector containing angular velocity/acceleration"""

TWIST_TYPE = np.ndarray
""" a 6D vector containing a 3D linear velocity and a 3D angular velocity"""
