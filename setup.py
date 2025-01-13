import setuptools
from setuptools import find_packages

setuptools.setup(
    name="mujoco-sim",
    version="0.0.1",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    description="",
    install_requires=[
        "dm_control",
        "numpy",
        "gymnasium",
        "spatialmath-python",
        "ur-analytic-ik",
        "robot_descriptions",
    ],  # old API for now.
    # extras_require={"sb3": ["stable-baselines3", "tqdm", "rich", "wandb", "tensorboard"]},
    packages=find_packages(),
)
