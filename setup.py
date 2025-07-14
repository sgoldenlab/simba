"""
SimBA (Simple Behavioral Analysis)
https://github.com/sgoldenlab/simba
Contributors.
https://github.com/sgoldenlab/simba#contributors-
Licensed under GNU Lesser General Public License v3.0
"""

import setuptools
import platform
from setuptools import setup, find_namespace_packages

REQUIREMENTS_PATH = 'requirements.txt'
ARM_REQUIREMENTS_PATH = 'requirements_arm.txt'
GPU_REQUIREMENTS_PATH = 'requirements_gpu.txt'

with open("docs/project_description.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(REQUIREMENTS_PATH, "r") as f:
    requirements = f.read().splitlines()

with open(ARM_REQUIREMENTS_PATH, "r") as f:
    arm_requirements = f.read().splitlines()

with open(GPU_REQUIREMENTS_PATH, "r") as f:
    gpu_requirements = f.read().splitlines()

# Setup configuration
setuptools.setup(
    name="simba_uw_tf_dev",
    version="3.4.2",
    author="Simon Nilsson, Jia Jie Choong, Sophia Hwang",
    author_email="sronilsson@gmail.com",
    description="Toolkit for computer classification and analysis of behaviors in experimental animals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgoldenlab/simba",
    install_requires=requirements,
    extras_require={'arm': [arm_requirements], "gpu": [gpu_requirements]},
    license='Modified BSD 3-Clause License',
    license_files=('LICENSE',),
    packages=setuptools.find_packages(exclude=["*.tests",
                                               "*.tests.*",
                                               "tests.*",
                                               "tests",
                                               "__pycache__",
                                               "pose_configurations_archive",
                                               "sandbox",
                                               "sandbox.*",
                                               ]),
    # packages=find_namespace_packages(include=["simba*"],
    #                                  exclude=["*.tests", "*.tests.*", "tests.*", "tests", "__pycache__", "__pycache__.*", "pose_configurations_archive", "pose_configurations_archive.*", "sandbox", "sandbox.*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={'console_scripts':['simba=simba.SimBA:main'],}
)



