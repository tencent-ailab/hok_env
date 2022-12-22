import os.path
import sys

from setuptools import setup, find_packages

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hok"))
from version import __version__

# Environment-specific dependencies.
extras = {}

setup(
    name="hok",
    version=__version__,
    description="Honor of Kings: A MOBA game environment for multi-agent reinforcement learning.",
    url="localhost",
    license="",
    packages=[package for package in find_packages() if package.startswith("gym")],
    zip_safe=False,
    install_requires=[],
    extras_require=extras,
    package_data={"gym": ["AILab"]},
    python_requires=">=3.6<=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
