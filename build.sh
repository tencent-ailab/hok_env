set -e
cpu_name=cpu
gpu_name=gpu

gpu_base_name=gpu_base
cpu_base_name=cpu_base
cpu_gamecore_base=cpu_gamecore_base

common_base_py37=common_base_py37
common_base_py38=common_base_py38
build_python_py38=build_python_py38

# common_base image for py37/py38
docker build -f ./dockerfile/dockerfile.base -t ${common_base_py38} --build-arg=PYTHON_VERSION=3.8.16 .
docker build -f ./dockerfile/dockerfile.base -t ${build_python_py38} --build-arg=PYTHON_VERSION=3.8.16 --target build_python .
docker build -f ./dockerfile/dockerfile.base -t ${common_base_py37} --build-arg=PYTHON_VERSION=3.7.16 .

# py37 for cpu, py38 for gpu
docker build -f ./dockerfile/dockerfile.base.cpu -t ${cpu_base_name} --build-arg=BASE_IMAGE=${common_base_py37} .
docker build -f ./dockerfile/dockerfile.base.gpu -t ${gpu_base_name} --build-arg=BASE_IMAGE=${common_base_py38} --build-arg=BUILD_PYTHON_IMAGE=${build_python_py38} .

# gamecore for cpu
docker build -f ./dockerfile/dockerfile.gamecore -t ${cpu_gamecore_base} --build-arg=BASE_IMAGE=${cpu_base_name} .

docker build -f ./dockerfile/dockerfile.dev -t ${cpu_name} --build-arg=BASE_IMAGE=${cpu_gamecore_base} .
docker build -f ./dockerfile/dockerfile.dev -t ${gpu_name} --build-arg=BASE_IMAGE=${gpu_base_name} .

# docker build -f ./dockerfile.base -t tencentailab/hok_env:base_python3.6_cuda10.0_cudnn7.6_nccl2.4.7_openmpi4.0.7_tf1.14_horovod0.16.4_$(git describe --dirty --always --tags | sed 's/-/_/g') .
# docker build -f ./dockerfile.dev -t tencentailab/hok_env:baseline_dev_1v1_$(git describe --dirty --always --tags | sed 's/-/_/g') .
# docker build -f ./dockerfile.cpu -t tencentailab/hok_env:baseline_cpu_1v1_$(git describe --dirty --always --tags | sed 's/-/_/g') .
