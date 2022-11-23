import numpy as np

POSITION_TYPE = np.ndarray
""" a 3D vector (x,y,z) [m] that represents the position of a frame in another frame;"""

ORIENTATION_TYPE = np.ndarray
""" a 4D vector containing a scalar-last quaternion (x,y,z,w)[radians] that represents the orientation of a frame in another frame"""
