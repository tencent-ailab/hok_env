#!/bin/bash

if [ $# -lt 1 ];then
    echo "usage $0 role master_ip log_dir"
    exit -1
fi

role=$1
master_ip=$2

LOG_DIR=${3-"/aiarena/logs/"}/model_pool
mkdir -p $LOG_DIR

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

MODEL_POOL_FILE_SAVE_PATH=${MODEL_POOL_FILE_SAVE_PATH:-"/mnt/ramdisk/model"}
mkdir -p ${MODEL_POOL_FILE_SAVE_PATH}
chmod +x ../bin/modelpool ../bin/modelpool_proxy

ln -sfnT $LOG_DIR $SCRIPT_DIR/../log

if [ $role = "cpu" ];then
   bash set_cpu_config.sh $master_ip
   cd ../bin && nohup ./modelpool -conf=../config/trpc_go.yaml > /dev/null 2>&1 &
   cd ../bin && nohup ./modelpool_proxy -fileSavePath=${MODEL_POOL_FILE_SAVE_PATH} > ${LOG_DIR}/modelpool_proxy.log 2>&1 &
fi

if [ $role = "gpu" ];then
   bash set_gpu_config.sh
   cd ../bin && nohup ./modelpool -conf=../config/trpc_go.yaml > /dev/null 2>&1 &
   cd ../bin && nohup ./modelpool_proxy -fileSavePath=${MODEL_POOL_FILE_SAVE_PATH} > ${LOG_DIR}/modelpool_proxy.log 2>&1 &
fi

