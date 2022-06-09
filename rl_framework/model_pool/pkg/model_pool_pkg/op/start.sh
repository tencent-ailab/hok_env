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

if [ $role = "cpu" ];then
   master_ip=`head -n 1 /code/gpu_code/script/gpu.iplist | awk '{print $1}'`
   bash set_cpu_config.sh $master_ip
   cd ../bin && nohup ./modelpool -conf=../config/trpc_go.yaml > /dev/null 2>&1 &
   cd ../bin && nohup ./modelpool_proxy -fileSavePath=/mnt/ramdisk/model > ../log/proxy.log 2>&1 &
fi

if [ $role = "gpu" ];then
   bash set_gpu_config.sh
   cd ../bin && nohup ./modelpool -conf=../config/trpc_go.yaml > /dev/null 2>&1 &
   cd ../bin && nohup ./modelpool_proxy -fileSavePath=/mnt/ramdisk/model > ../log/proxy.log 2>&1 &
fi
