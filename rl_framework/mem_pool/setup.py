from setuptools import setup
from setuptools import find_packages

PROJECT_NAME = "rl-framework-mem-pool"
_VERSION = "1.0.0"

setup(
    name=PROJECT_NAME,
    version=_VERSION,
    packages=find_packages(),
    description="rl-framework-mem-pool",
    long_description="rl-framework-mem-pool",
    license="Apache 2.0",
    keywords="rl-framework game ai training framework - mem_pool",
    install_requires=["lz4"],
)
