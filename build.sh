docker build -f ./dockerfile.base -t mirrors.tencent.com/king-kaiwu/baseline:base_python3.6_cuda10.0_cudnn7.6_nccl2.4.7_openmpi4.0.7_tf1.14_horovod0.16.4 .
docker build -f ./dockerfile.dev -t mirrors.tencent.com/king-kaiwu/baseline:dev_1v1_20220608 .
