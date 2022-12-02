from setuptools import setup
from setuptools import find_packages

PROJECT_NAME = "rl-framework-learner"
_VERSION = "1.0.0"

require_list = []

setup(
    name=PROJECT_NAME,
    version=_VERSION,
    packages=find_packages(),
    description="rl-framework-learner",
    long_description="rl-framework-learner",
    license="Apache 2.0",
    keywords="rl-framework game ai training framework - learner",
    install_requires=require_list,
)
