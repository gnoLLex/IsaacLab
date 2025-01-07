# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'omni.isaac.lab_tasks' python package."""

import itertools
import os
import platform
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "torch==2.4.0",
    "theseus",
    "torchkin",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Extra dependencies for RL agents
EXTRAS_REQUIRE = {
    "sb3": ["stable-baselines3>=2.1"],
    "skrl": ["skrl>=1.3.0"],
    "rl-games": ["rl-games==1.6.1", "gym"],  # rl-games still needs gym :(
    "rsl-rl": ["rsl-rl@git+https://github.com/leggedrobotics/rsl_rl.git"],
    "robomimic": [],
}


# Installation operation
setup(
    name="robot_utils",
    author="Lucas Engelmann",
    maintainer="Lucas Engelmann",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    extras_require=EXTRAS_REQUIRE,
    packages=["omni.isaac.lab_tasks"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.2.0",
        "Isaac Sim :: 4.1.0",
    ],
    zip_safe=False,
)
