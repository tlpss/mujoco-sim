"""
idea of this env is to do sim2real on RGB(-D) observations
for learning closed-loop planar pushing of various items towards a target location (w/constant material)
kind of like procgen generalization but for manipulation.

and to test continuous vs discrete vs implicit (spatial actions)

and handheld camera vs static camera?

requires:
- a robot
- a workspace enlarging and self-collision avoiding EEF for dealing with UR3e constraints! (e.g Z shaped that can rotate around)
- an arena that is varied but should include environments similar to the real-world table
- random objects to push around
- an admittance controller to make the whole thing safe?

- if robot pushes something out of its workspace, the episode terminates with a penalty
    so don't babysit it by allowing only actions that would keep the object in the workspace..
- varying number of objects in the scene etc.
"""
