#!/bin/bash

function log() {
    now=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$now] $1"
}

############################
MODEL_POOL_PKG_DIR=${MODEL_POOL_PKG_DIR:-"/rl_framework/model_pool/pkg/model_pool_pkg/"}
LOG_DIR="/aiarena/logs/actor/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input_learner_list=${input_learner_list:-"${SCRIPT_DIR}/learner.iplist"}
export learner_list=${learner_list:-"${SCRIPT_DIR}/learner.iplist.new"}

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
    master_ip=$(head -n 1 ${learner_list} | awk '{print $1}')
    cd ${MODEL_POOL_PKG_DIR}/op && bash stop.sh && bash start.sh cpu $master_ip $LOG_DIR
fi

if [ "$DEPLOY_GAMECORE" = "1" ]; then
    sh /rl_framework/remote-gc-server/start_gamecore_server.sh
fi

############################
log "start actor"
bash ${SCRIPT_DIR}/kill.sh
nohup bash ${SCRIPT_DIR}/monitor_actor.sh >>${LOG_DIR}/monitor.log 2>&1 &
