#!/bin/bash
#this starts rl learner for training
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/code/code/gpu_code/learner/code/shm_lib"
export SLOW_TIME=0.03
mount -o size=200M -o nr_inodes=1000000 -o noatime,nodiratime -o remount /dev/shm

if [ $# -lt 2 ];then
    echo "usage $0 training_type game_name(atari/kinghonour) [is_chief_node]"
    exit
fi

training_type=$1
game_name=$2
is_chief_node="1"
if [ $# = 3 ];then
    is_chief_node=$3
fi
echo "is_chief_node: ", $is_chief_node

type1="async"
type2="sync"
#sleep 120

if [[ ${is_chief_node} == "1" ]];then
    cd /code/code/gpu_code/learner/
    nohup bash ./code/run_multi.sh >> /code/logs/gpu_log/start.log  &
    if [ "${training_type}"x == "async"x ]
    then
        echo "start model_pool"
        cd /rl_framework/model_pool/pkg/model_pool_pkg/op && bash stop.sh && bash start.sh gpu
    elif [ "${training_type}"x == "sync"x ]
    then
        sleep 3
        nohup python ./code/server.py > log/server.log 2>&1 &
        nohup sh ./model/delete.sh > log/del.log 2>&1 &
    else
        echo "training type does not exists", $training_type
    fi
fi
