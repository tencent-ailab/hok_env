function log(){
  now=`date +"%Y-%m-%d %H:%M:%S"`
  echo "[$now] $1"
}

#!/bin/bash
function init(){
  log "init dir"
  rm -rf /code/model_bkup /code/logs/cpu_log /code/logs/gpu_log /code/code/gpu_code/send_model/model
  mkdir -p /code/model_bkup /code/logs/cpu_log /code/logs/gpu_log /code/code/gpu_code/send_model/model

  sed -i "s/mem_pool_num=.*/mem_pool_num=$CARD_NUM/g" /code/code/gpu_code/script/config.conf
  echo "task_id=${TASK_ID}" >> /code/code/gpu_code/script/config.conf
  echo "task_uuid=${TASK_ID}" >> /code/code/gpu_code/script/config.conf

  cd /code/code/gpu_code/script/
  gpu_iplist=./gpu.iplist
  cpu_iplist=./cpu.iplist
  touch cpu.iplist

  cp ./cpu.iplist /code/code/gpu_code/learner/tool/
  cp ./gpu.iplist /code/code/gpu_code/learner/tool/
  cp ./config.conf /code/code/gpu_code/learner/tool/

  log "start run set_gpu.sh"
  cd /code/code/gpu_code/learner/tool/ && bash set_gpu.sh
}

function start() {
  IS_CHIEF_NODE='1'
  cd /code/code/gpu_code/learner/; bash kill.sh; bash clean.sh

  cd /code/code/gpu_code/learner/
  training_type=async
  game_name=1v1
  nohup bash start.sh $training_type $game_name $IS_CHIEF_NODE > /code/logs/gpu_log/start.log 2>&1 &
  log "start rl learner"

  nohup influxdb_exporter --web.listen-address=":8086"  --udp.bind-address=":8086" > /dev/null 2>&1 &
  log "start monitor"
}

init
start

