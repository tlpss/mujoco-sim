# Mujoco-sim
Learning environments for robotic manipulation using MuJoCo with the dm_control package. (this repo also serves as project for learning to use both myself, so not everything is implemented in the best/ most canonical way).

# Tasks
## Robot Tasks
All these tasks have both dense and sparse rewards, and both visual and state observations.


<img align="right" width="100" height="100" src="doc/robot-reach.png">

**Reach**
The agent has to reach a target location in the Euclidean space by providing delta steps for the position (xyz).

<br>
<br>
<br>

<img align="left" width="100" height="100" src="doc/robot-planar-push.png">

**Planar Push**
The agent controls delta xy coordinates of a cylindrical EEF and has to push a configurable number of objects to a target location (indicated by a white disc).

<br>
<br>






# repo layout

```
/mujoco_sim
    /entities           # all phyiscal objects in the environments, using the composer.Entity abstraction
        /arenas         # roots of the entity tree, contain the 'robot setup'
        /robots         # actual robots
        /eef            # grippers etc.
        /props          # non-actuated elements
    /environments
        /tasks          # implements the actual learning tasks
        dmc2gym.py      # converts the DMC environment interface to a gym interface for interacting with most RL frameworks
    /mjcf               # contains the mjcf xml files for all the entities

```
# installation

- `git clone`
- `git submodules update --init`
- `conda env create -f environment.yaml`
- `pip install -e ur_ikfast/`
- `pip install -e airo-core/`

for learning:
- `pip install -e .[sb3]`

# Dependencies and information sources
## Dependencies
- mujoco_menagerie: library of MJCF models
- ur_ikfast: python ikfast wrapper for UR robot class