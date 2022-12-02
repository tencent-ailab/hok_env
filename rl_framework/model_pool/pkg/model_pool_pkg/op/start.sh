#!/bin/bash

if [ $# -lt 1 ];then
    echo "usage $0 role"
    exit -1
fi

role=$1

if [ -d "../log" ]; then
    rm -r ../log
fi
mkdir ../log

if [ ! -d /mnt/ramdisk/model ];then
    mkdir -p /mnt/ramdisk/model
fi

chmod +x ../bin/modelpool ../bin/modelpool_proxy

if [ $role = "cpu" ];then
   master_ip=`head -n 1 /code/code/cpu_code/script/gpu.iplist.new | awk '{print $1}'`
   bash set_cpu_config.sh $master_ip
   cd ../bin && nohup ./modelpool -conf=../config/trpc_go.yaml > /dev/null 2>&1 &
   cd ../bin && nohup ./modelpool_proxy -fileSavePath=/mnt/ramdisk/model > /code/logs/modelpool_proxy.log 2>&1 &
fi

if [ $role = "gpu" ];then
   bash set_gpu_config.sh
   cd ../bin && nohup ./modelpool -conf=../config/trpc_go.yaml > /dev/null 2>&1 &
   cd ../bin && nohup ./modelpool_proxy -fileSavePath=/mnt/ramdisk/model > /code/logs/modelpool_proxy.log 2>&1 &
fi

sleep 20
cp -r /rl_framework/model_pool/pkg/model_pool_pkg/log/modelpool.log /code/logs/ 
