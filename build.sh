docker build -f ./dockerfile.base -t tencentailab/hok_env:base_python3.6_cuda10.0_cudnn7.6_nccl2.4.7_openmpi4.0.7_tf1.14_horovod0.16.4_$(git describe --dirty --always --tags | sed 's/-/_/g') .
docker build -f ./dockerfile.dev -t tencentailab/hok_env:baseline_dev_1v1_$(git describe --dirty --always --tags | sed 's/-/_/g') .
