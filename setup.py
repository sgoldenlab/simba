"""
SimBA (Simple Behavioral Analysis)
https://github.com/sgoldenlab/simba
Contributors.
https://github.com/sgoldenlab/simba#contributors-
Licensed under GNU Lesser General Public License v3.0
"""

import setuptools
import platform

REQUIREMENTS_PATH = 'requirements.txt'
ARM_REQUIREMENTS_PATH = 'requirements_arm.txt'

with open("docs/project_description.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(REQUIREMENTS_PATH, "r") as f:
    requirements = f.read().splitlines()

with open(ARM_REQUIREMENTS_PATH, "r") as f:
    arm_requirements = f.read().splitlines()

# Setup configuration
setuptools.setup(
    name="Simba-UW-tf-dev",
    version="2.0.8",
    author="Simon Nilsson, Jia Jie Choong, Sophia Hwang",
    author_email="sronilsson@gmail.com",
    description="Toolkit for computer classification and analysis of behaviors in experimental animals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgoldenlab/simba",
    install_requires=requirements,
    extras_require={'arm': [arm_requirements]},
    license='GNU General Public License v3 (GPLv3)',
    packages=setuptools.find_packages(exclude=["*.tests",
                                               "*.tests.*",
                                               "tests.*",
                                               "tests",
                                               "__pycache__",
                                               "pose_configurations_archive"]),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points={'console_scripts':['simba=simba.SimBA:main'],}
)