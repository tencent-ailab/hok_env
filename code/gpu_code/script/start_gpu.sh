#!/bin/bash
task_name=test_auto
training_type=async
steps=157
game_name=1v1
env_name=1v1
mem_pool_num=1
task_id=47981
task_uuid=a904743a68ab4a9982f80215165d770e
card_num=1
export USER=root
if [ ! $card_num ]; then
  echo "card_num is NULL!" 
  exit
fi
sed -i "s/mem_pool_num=.*/mem_pool_num=$card_num/g" /code/gpu_code/script/config.conf
echo "task_id=${TASK_ID}" >> /code/gpu_code/script/config.conf
echo "task_uuid=${TASK_UUID}" >> /code/gpu_code/script/config.conf

cd /code/gpu_code/script/
. ./config.conf

gpu_iplist=./gpu.iplist
cpu_iplist=./cpu.iplist
touch cpu.iplist

cp ./cpu.iplist /code/gpu_code/learner/tool/
cp ./gpu.iplist /code/gpu_code/learner/tool/
cp ./config.conf /code/gpu_code/learner/tool/

sh start_monitor.sh
echo "start run set_gpu.sh"
cd /code/gpu_code/learner/tool/ && bash set_gpu.sh

IS_CHIEF_NODE='1'
cd /code/gpu_code/learner/ ; bash kill.sh; bash clean.sh

sleep 15
cd /code/gpu_code/learner/
nohup bash start.sh $training_type $game_name $IS_CHIEF_NODE > start.log 2>&1 &
echo "Start RL learner"
