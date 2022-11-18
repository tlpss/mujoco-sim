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
    ],
    packages=find_packages(),
)
