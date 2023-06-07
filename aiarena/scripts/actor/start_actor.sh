#!/bin/bash

function log(){
  now=`date +"%Y-%m-%d %H:%M:%S"`
  echo "[$now] $1"
}

############################
CPU_NUM=${CPU_NUM:-"2"}
mem_pool_num=${mem_pool_num:-"1"}
MODEL_POOL_PKG_DIR=${MODEL_POOL_PKG_DIR:-"/rl_framework/model_pool/pkg/model_pool_pkg/"}
LOG_DIR="/aiarena/logs/actor/"
ACTOR_CODE_DIR=${ACTOR_CODE_DIR:-"/aiarena/code/actor"}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
input_learner_list=${input_learner_list:-"${SCRIPT_DIR}/learner.iplist"}
learner_list=${learner_list:-"${SCRIPT_DIR}/learner.iplist.new"}

mkdir -p ${LOG_DIR}

############################
log "parse ip list"
cd ${SCRIPT_DIR}
python parse_iplist.py ${input_learner_list} ${learner_list} 1

############################
# 等待的 learner 启动，通过探测 model pool 端口实现
ip=$(cat ${learner_list} | awk '{print $1}' | sed -n '1p')
log "learner ip: $ip"
while true; do
    code=$(curl -sIL -w "%{http_code}\n" -o /dev/null http://$ip:10016)
    if [ $code -gt 200 ]; then
        log "learner is ok"
        break
    fi
    log "learner is not ok, wait for ready"
    sleep 1
done

############################
if [ -z "$NO_ACTOR_MODEL_POOL" ]; then
  log "start model_pool"
  cd ${MODEL_POOL_PKG_DIR}/op; bash stop.sh; bash start.sh cpu ${learner_list} ${LOG_DIR}
fi

############################
log "parse mem pool"
let end_num=mem_pool_num-1
idx=0
mem_pool_addr=""
while read ip user port gpu_card_num;do
    for i in `seq 0 $end_num`
    do
        let port=35200+$i
        log "mem_pool_$idx $ip:$port"
        if [ $idx -eq 0 ]
        then
            mem_pool_addr="$ip:$port"
        else
            mem_pool_addr="${mem_pool_addr};$ip:$port"
        fi
        let idx+=1
    done
done < $learner_list
log ${mem_pool_addr}

monitor_server_addr=`cat ${learner_list} |head -n 1|awk '{print $1}'`:8086
log "monitor_server_addr: $monitor_server_addr"

############################
MAX_EPISODE=${MAX_EPISODE-"-1"}
log "start actor"
bash ${SCRIPT_DIR}/kill.sh
cd ${ACTOR_CODE_DIR}
let actor_num=CPU_NUM-1
for i in $(seq 0 $actor_num); do
    nohup python entry.py --actor_id=$i \
                          --mem_pool_addr=$mem_pool_addr \
                          --model_pool_addr="localhost:10016" \
                          --max_episode=${MAX_EPISODE} \
                          --monitor_server_addr=${monitor_server_addr} \
                          >> ${LOG_DIR}/actor_$i.log 2>&1 &
done;
