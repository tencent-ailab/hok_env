function log() {
    now=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$now] $1"
}

LEARNER_DIR="/aiarena/code/learner/"
LOG_DIR="/aiarena/logs/learner/"
MODEL_POOL_PKG_DIR=${MODEL_POOL_PKG_DIR:-"/rl_framework/model_pool/pkg/model_pool_pkg/"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

input_learner_list=${input_learner_list:-"${SCRIPT_DIR}/learner.iplist"}
learner_list=${learner_list:-"${SCRIPT_DIR}/learner.iplist.new"}

mkdir -p ${LOG_DIR}

##############################
log "stop learner"
cd $SCRIPT_DIR && bash kill.sh

############################
log "parse ip list"
cd ${SCRIPT_DIR}
python parse_iplist.py ${input_learner_list} ${learner_list} 1

##############################
log "start model_pool"
master_ip=$(head -n 1 ${learner_list} | awk '{print $1}')
cd ${MODEL_POOL_PKG_DIR}/op && bash stop.sh && bash start.sh gpu $master_ip $LOG_DIR

##############################
log "start monitor"
cd $SCRIPT_DIR && sh start_monitor.sh

##############################
node_list=$(cat ${learner_list} | awk '{print $1":"$4}' | xargs | sed 's| |,|g')
node_num=$(cat ${learner_list} | wc -l)
log "node_list: ${node_list}"
log "node_num: ${node_num}"

##############################
learner_num=$(cat $learner_list | awk '{print $4}' | awk '{sum+=$1} END {print sum}')
proc_per_node=$((${learner_num}/${node_num}))
NET_CARD_NAME=${NET_CARD_NAME:-"eth0"}
log "learner_num: ${learner_num}"
log "proc_per_node: ${proc_per_node}"
log "NET_CARD_NAME: ${NET_CARD_NAME}"

##############################
mem_pool_num=$(head -n 1 ${learner_list} | awk '{print $4}')
mem_pool_list='[35200'
for ((i = 1; i < mem_pool_num; i++)); do
    mem_pool_port=$((35200 + i))
    mem_pool_list=$mem_pool_list','$mem_pool_port
done
mem_pool_list=${mem_pool_list}']'
sed -i "s/ports =.*/ports = ${mem_pool_list}/g" $LEARNER_DIR/config/common.conf
log "mem_pool_list: ${mem_pool_list}"

function run_learner_mpi() {
    log "start learner (mpirun)"
    cd ${LEARNER_DIR}
    export PATH=/data/opt/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/opt/ibutils/bin:/root/bin:$PATH
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
    nohup mpirun --allow-run-as-root -np ${learner_num} -H ${node_list} -bind-to none -map-by slot \
        -mca plm_rsh_args "-p 36001" \
        -x MASTER_ADDR=${master_ip} \
        -x MASTER_PORT=23333 \
        -x NCCL_IB_DISABLE=1 \
        -x NCCL_SOCKET_IFNAME=$NET_CARD_NAME \
        -x NCCL_DEBUG=INFO \
        -x LD_LIBRARY_PATH python3 train.py >>${LOG_DIR}/train.log 2>&1 &
}

function run_learner_ddp() {
    log "start learner (torchrun)"
    cd ${LEARNER_DIR}
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=$NET_CARD_NAME
    export NCCL_DEBUG=INFO
    nohup torchrun \
    --nnodes ${node_num} \
    --nproc_per_node=${proc_per_node} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${master_ip} \
    train.py >> ${LOG_DIR}/train.log 2>&1 &
}

use_ddp=0
if grep -q "backend\s*=\s*pytorch" /aiarena/code/learner/config/common.conf; then
    # Check if the distributed_backend is set to ddp
    if grep -q "distributed_backend\s*=\s*ddp" /aiarena/code/learner/config/common.conf; then
        use_ddp=1
    fi
fi

function run_learner() {
    if [[ ${use_ddp} -eq 1 ]]; then
        run_learner_ddp
    else
        run_learner_mpi
    fi
}

##############################
if [[ $(cat ${learner_list} | wc -l) -ne 1 ]] && [[ ${use_ddp} -eq 0 ]]; then
    # config sshd
    # 修改密码并启动sshd
    rm -rf /root/.ssh/authorized_keys && ssh-keygen -t rsa -N '' -f /root/.ssh/id_rsa <<<y
    NEW_PASSWORD=${NEW_PASSWORD-${TASK_UUID-'random_pass_uuid'}}
    echo "root:${NEW_PASSWORD}" | chpasswd

    mkdir -p /run/sshd
    /usr/sbin/sshd

    # 等待sshd
    while true; do
        continue_test=0
        while read line; do
            line=$(echo $line | awk '{print $1}')
            echo "test $line"
            echo -e '\035' | telnet $line 36001 || continue_test=1
        done <${learner_list}

        if [[ $continue_test -eq 0 ]]; then
            echo "all test done"
            break
        fi
        sleep 1
    done

    cat ${learner_list} | awk '{print $1}' | while read line; do
        ./ssh-copy-id.expect $line $NEW_PASSWORD
    done

    if [[ "$(hostname -I | awk '{print $1}')" == "${master_ip}" ]]; then
        run_learner
    fi

else
    run_learner
fi

##############################
# clean ckpt and backup with build_code.sh
nohup python3 /rl_framework/send_model/check_and_send_checkpoint.py >>$LOG_DIR/send.log 2>&1 &
