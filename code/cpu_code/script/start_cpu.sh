#!/bin/bash
CPU_NUM=8
card_num=1
task_name=test_auto
training_type=async
steps=157
game_name=1v1
env_name=1v1
mem_pool_num=1
task_id=47981
task_uuid=a904743a68ab4a9982f80215165d770e
export USER=root
sed -i "s/mem_pool_num=.*/mem_pool_num=$card_num/g" /code/gpu_code/script/config.conf
# Dump var in config.conf
echo task_id=$TASK_ID >> /code/cpu_code/script/config.conf
echo task_uuid=$TASK_UUID >> /code/cpu_code/script/config.conf

# Deal with params
cd /code/cpu_code/script
. ./config.conf
gpu_iplist=/code/cpu_code/cpu_code/gpu.iplist.new
cpu_iplist=/code/cpu_code/script/cpu.iplist
echo "start setup param"
bash setup_param.sh $gpu_iplist $cpu_iplist $mem_pool_num $task_id $task_uuid

# Setup and start actor

echo "start set up gamecore"
cd /code/cpu_code/script/; bash deploy_gamecore.sh

# Start actor
echo "start actor"
sleep 30
cd /code/cpu_code/actor/; bash kill.sh; bash start.sh $CPU_NUM
echo 'bash setup_param.sh '$gpu_iplist' '$cpu_iplist' '$mem_pool_num' '$task_id' '$task_uuid