#!/bin/bash
function log() {
    now=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$now] $1"
}
max_test_time=${max_test_time-"300"}

common_conf_file="/aiarena/code/learner/config/common.conf"
LEARNER_LOG=/aiarena/logs/learner/train.log
ACTOR_LOG=/aiarena/logs/actor/actor_0.log

function init() {
    # 删除旧的日志目录, 避免软链失败
    rm -rf /aiarena/code/actor/log
    rm -rf /aiarena/code/learner/log

    conf_backup_dir=$1
    # 已经存在, 说明上次测试被中断, 对应的配置可能是旧的
    if [[ ! -e "${conf_backup_dir}/common.conf" ]]; then
        cp ${common_conf_file} ${conf_backup_dir}
    fi

    sed -i "s|store_max_sample.*|store_max_sample = 20|g" ${common_conf_file}
    sed -i "s|display_every.*|display_every = 1|g" ${common_conf_file}
    sed -i "s|save_model_steps.*|save_model_steps = 2|g" ${common_conf_file}
    sed -i "s|max_steps.*|max_steps = 5|g" ${common_conf_file}
    sed -i "s|batch_size.*|batch_size = 1|g" ${common_conf_file}
    sed -i "s|use_xla.*|use_xla = False|g" ${common_conf_file}
}

function uninit() {
    conf_backup_dir=$1
    cp ${conf_backup_dir}/common.conf ${common_conf_file}
    rm -r ${conf_backup_dir}

    # 删除软链, 避免windows下打包失败
    rm -rf /aiarena/code/actor/log
    rm -rf /aiarena/code/learner/log

    mkdir -p /aiarena/logs/actor/
    mv /aiarena/code/actor/GameAiMgr_*.txt /aiarena/logs/actor/
}

# 等待的 learner 启动，通过探测 model pool 端口实现
function wait_learner() {
    while true; do
        grep "init finished" ${LEARNER_LOG} 1>/dev/null 2>&1 && break
        log "learner is not ok, wait"
        sleep 1
    done
    log "learner is ok"
}

function wait_actor() {
    while true; do
        grep "Start a new game" ${ACTOR_LOG} 1>/dev/null 2>&1 && log "actor is ok" && break
        if [[ $(ps -e f | grep train.py | grep -v grep | wc -l) -eq 0 ]]; then
            break
        fi
        log "actor is not ok, wait"
        sleep 1
    done
}

start_time=$(date +%s)
rm -rf /aiarena/logs
rm -rf /aiarena/checkpoints

# 允许外部修改actor_num, 多learner测试的场景需要多个actor进程以给各个learner发送数据
export ACTOR_NUM=${ACTOR_NUM-"1"}
# export MAX_EPISODE=${MAX_EPISODE-"1"}
export ACTOR_RUNTIME_ID_PREFIX=${ACTOR_RUNTIME_ID_PREFIX-"start-test"}
export ACTOR_PORT_BEGIN=${ACTOR_PORT_BEGIN-"35100"}
export MAX_FRAME_NUM=${MAX_FRAME_NUM-"1000"}
export USE_XLA=${USE_XLA-"0"}

conf_backup_dir=/aiarena/conf_backup/
mkdir -p ${conf_backup_dir}
init ${conf_backup_dir}
log "---------------------------start testing----------------------------"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sh $SCRIPT_DIR/start_dev.sh

wait_learner
wait_actor

# Code testing
ts_tag=-1
while [ $(($(date +%s) - start_time)) -lt $max_test_time ]; do
    if [[ -e "/aiarena/logs/learner/loss.txt" && $(ls -l /aiarena/logs/learner/loss.txt | awk '{print $5}') -ne 0 ]]; then
        log "time cost: $(($(date +%s) - start_time))"
        log "---------------------------testing success----------------------------"
        ts_tag=0
        break
    elif [[ $(ps -e f | grep train.py | grep -v grep | wc -l) -eq 0 ]]; then
        log "time cost: $(($(date +%s) - start_time))"
        log "---------------------------testing fail------------------------------"
        ts_tag=1
        break
    elif [[ $(ps -e f | grep entry.py | grep -v grep | wc -l) -eq 0 ]]; then
        log "time cost: $(($(date +%s) - start_time))"
        log "---------------------------testing fail------------------------------"
        ts_tag=1
        break
    fi
    sleep 1
done

if [ $(($(date +%s) - start_time)) -ge $max_test_time ]; then
    log "time cost: $(($(date +%s) - start_time))"
    log "---------------------------testing fail------------------------------"
    ts_tag=2
fi

sh $SCRIPT_DIR/stop_dev.sh >/dev/null 2>&1
uninit ${conf_backup_dir}
GAMECORE_SERVER_ADDR=${GAMECORE_SERVER_ADDR-"127.0.0.1:23432"}
let actor_end=ACTOR_NUM-1
for i in $(seq 0 $actor_end); do
    curl --max-time 30 http://$GAMECORE_SERVER_ADDR/v2/stopGame -H 'Content-Type: application/json' -d '{"runtime_id": "'${ACTOR_RUNTIME_ID_PREFIX}'-'${i}'"}'
done

log "---------------------------testing finish----------------------------"
log ------------------------------"$ts_tag"-------------------------------

if [[ $ts_tag -eq 0 ]]; then
    exit 0
elif [[ $ts_tag -eq 1 ]]; then
    exit 1
elif [[ $ts_tag -eq 2 ]]; then
    exit 203
fi
