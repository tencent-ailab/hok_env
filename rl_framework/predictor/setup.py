from setuptools import setup
from setuptools import find_packages

PROJECT_NAME = "rl-framework-predictor"
_VERSION = "1.0.0"

require_list = []
cpu_list = []
gpu_list = []

setup(
    name=PROJECT_NAME,
    version=_VERSION,
    packages=find_packages(),
    description="rl-framework-predictor",
    long_description="rl-framework-predictor",
    license="Apache 2.0",
    keywords="rl-framework game ai training framework - predictor",
    install_requires=require_list,
    extras_require={
        "cpu": cpu_list,
        "gpu": gpu_list,
    },
)
