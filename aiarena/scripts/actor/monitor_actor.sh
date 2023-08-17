function log() {
    now=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$now] $1"
}

############################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
learner_list=${learner_list:-"${SCRIPT_DIR}/learner.iplist.new"}

log "parse mem pool"
idx=0
mem_pool_addr=""
while read ip user port gpu_card_num; do
    let end_num=gpu_card_num-1
    for i in $(seq 0 $end_num); do
        let port=35200+i
        log "mem_pool_$idx $ip:$port"
        if [ $idx -eq 0 ]; then
            mem_pool_addr="$ip:$port"
        else
            mem_pool_addr="${mem_pool_addr};$ip:$port"
        fi
        let idx+=1
    done
done <$learner_list
log ${mem_pool_addr}

############################

monitor_server_addr=$(cat ${learner_list} | head -n 1 | awk '{print $1}'):8086
log "monitor_server_addr: $monitor_server_addr"

############################

ACTOR_CODE_DIR=${ACTOR_CODE_DIR:-"/aiarena/code/actor"}
ACTOR_NUM=${ACTOR_NUM:-${CPU_NUM:-"1"}}
MAX_EPISODE=${MAX_EPISODE-"-1"}
LOG_DIR="/aiarena/logs/actor/"
mkdir -p $LOG_DIR

let actor_end=ACTOR_NUM-1
while [ "1" == "1" ]; do
    cd ${ACTOR_CODE_DIR}
    for i in $(seq 0 $actor_end); do
        actor_cnt=$(ps -elf | grep "python entry.py --actor_id=$i " | grep -v grep | wc -l)
        log "actor_id:$i actor_cnt:$actor_cnt"
        if [ $actor_cnt -lt 1 ]; then
            log "restart actor_id:$i"
            nohup python entry.py --actor_id=$i \
                --mem_pool_addr=$mem_pool_addr \
                --model_pool_addr="localhost:10016" \
                --max_episode=${MAX_EPISODE} \
                --monitor_server_addr=${monitor_server_addr} \
                >>${LOG_DIR}/actor_$i.log 2>&1 &
            sleep 1
        fi
    done # for

    sleep 30

done # while
