from setuptools import setup
from setuptools import find_packages

PROJECT_NAME = "rl-framework-common"
_VERSION = "1.0.0"

setup(
    name=PROJECT_NAME,
    version=_VERSION,
    packages=find_packages(),
    description="rl-framework-common",
    long_description="rl-framework-common",
    license="Apache 2.0",
    keywords="rl-framework game ai training framework - common",
    install_requires=["pyzmq", "loguru"],
)
