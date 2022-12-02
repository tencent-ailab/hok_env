#!/bin/bash
if [ $# -lt 1 ];then
    echo "usage $0 actor_num"
    exit
fi
bash ./kill.sh
actor_num=$1

if [ ! -d "./log" ]; then
    mkdir log 
else
    rm -r ./log/ && mkdir log 
fi

if [ -f "*.log" ]; then
    rm *.log
fi

mem_pool_config=config/mem_pool.host_list
if [ ! -f $mem_pool_config ];then
    echo "mem_pool_config doesn't exist: $mem_pool_config"
    exit 0
fi
mem_pool_addr=`cat config/mem_pool.host_list |xargs |sed 's/ /;/g'`
echo $mem_pool_addr

#echo "decompress freeze_model!"
#tar -xf /root/freeze_model.tar
#mv checkpoints_* freeze_model

cpu_list=`cat /sys/fs/cgroup/cpuset/cpuset.cpus`

cd code;
let actor_num=$actor_num-1
for i in $(seq 0 $actor_num); do
#   use taskset to bind cpu
    nohup python entry.py --actor_id=$i \
                          --mem_pool_addr=$mem_pool_addr \
                          --model_pool_addr="localhost:10016" \
                          --thread_num=1 \
                          --game_log_path "/code/logs/game_log" \
                          >> /code/logs/cpu_log/actor_$i.log 2>&1 &

done;
