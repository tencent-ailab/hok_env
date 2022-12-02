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

function init(){
  rm -rf  /code/logs/cpu_log /code/logs/game_log
  mkdir -p /code/logs/cpu_log /code/logs/game_log

  # set cpu.iplist
  echo "$POD_IP root 36001 $CPU_NUM" > /code/code/cpu_code/script/cpu.iplist

  # set config conf
  sed -i "s/mem_pool_num=.*/mem_pool_num=$CARD_NUM/g" /code/code/gpu_code/script/config.conf
  sed -i "s/mem_pool_num=.*/mem_pool_num=$CARD_NUM/g" /code/code/cpu_code/script/config.conf
  task_uuid=$TASK_ID
  echo task_id=$TASK_ID >> /code/code/cpu_code/script/config.conf
  echo task_uuid=$task_uuid >> /code/code/cpu_code/script/config.conf

  echo "start setup param"
  cd /code/code/cpu_code/script
  python parse_iplist.py /code/code/cpu_code/script/gpu.iplist /code/code/cpu_code/script/gpu.iplist.new 1
  bash /code/code/cpu_code/script/setup_param.sh \
                      /code/code/cpu_code/script/gpu.iplist.new \
                      /code/code/cpu_code/script/cpu.iplist \
                      $CARD_NUM $TASK_ID $task_uuid
}

# 等待的 learner 启动，通过探测 model pool 端口实现
function wait_learner() {
  ip=$(cat /code/code/cpu_code/script/gpu.iplist | awk '{print $1}' | sed -n '1p')
  echo "learner ip: $ip"
  while true; do
      code=$(curl -sIL -w "%{http_code}\n" -o /dev/null http://$ip:10016)
      if [ $code -gt 200 ]; then
          echo "learner is ok"
          break
      fi
      echo "learner is not ok, wait for ready"
      sleep 1
  done
}


function start(){
  echo "start model_pool"
  if [ ` ps -ef |grep -v "grep" | grep -c "start_gpu.sh" ` -gt 0 ];then
    train_type="gpu"
  else
    train_type="cpu"
  fi
  if [ -z "$NO_MODEL_POOL" ]; then
    cd /rl_framework/model_pool/pkg/model_pool_pkg/op; bash stop.sh; bash start.sh $train_type
    sleep 10
  fi

  echo "start actor"
  cd /code/code/cpu_code/actor/; bash start.sh $CPU_NUM
}

function start_monitor(){
  bash /code/code/cpu_code/script/actor_monitor.sh
}

wait_learner
init
start
start_monitor
