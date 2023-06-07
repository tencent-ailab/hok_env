from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hok"))
from version import __version__

# Environment-specific dependencies.
extras = {}

setup(
    name="hok",
    version=__version__,
    description="Honor of Kings: A MOBA game environment for multi-agent reinforcement learning.",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.6, <3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.7',
    ],
)
