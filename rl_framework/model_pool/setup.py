from setuptools import setup
from setuptools import find_packages

PROJECT_NAME = "rl-framework-model-pool"
_VERSION = "1.0.0"

require_list = ["timeout_decorator"]

setup(
    name=PROJECT_NAME,
    version=_VERSION,
    packages=find_packages(),
    description="rl-framework-model-pool",
    long_description="rl-framework-model-pool",
    license="Apache 2.0",
    keywords="rl-framework game ai training framework - model_pool",
    install_requires=require_list,
)
