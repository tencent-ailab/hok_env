function log(){
  now=`date +"%Y-%m-%d %H:%M:%S"`
  echo "[$now] $1"
}

LEARNER_DIR="/aiarena/code/learner/"
LOG_DIR="/aiarena/logs/learner/"
MODEL_POOL_PKG_DIR=${MODEL_POOL_PKG_DIR:-"/rl_framework/model_pool/pkg/model_pool_pkg/"}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
learner_iplist=$SCRIPT_DIR/learner.iplist

mkdir -p ${LOG_DIR}

##############################
log "stop learner"
cd $SCRIPT_DIR && bash kill.sh;

##############################
log "start model_pool"
cd ${MODEL_POOL_PKG_DIR}/op && bash stop.sh && bash start.sh gpu $learner_iplist $LOG_DIR

##############################
log "start monitor"
cd $SCRIPT_DIR && sh start_monitor.sh

##############################
node_list=`cat learner.iplist | awk '{print $1":"$4}'|xargs|sed 's| |;|g'`
log "node_list: ${node_list}"

##############################
gpu_num=`cat $learner_iplist | awk '{print $4}' | awk '{sum+=$1} END {print sum}'`
NET_CARD_NAME=${NET_CARD_NAME:-"eth0"}
log "gpu_num: ${gpu_num}"
log "NET_CARD_NAME: ${NET_CARD_NAME}"

##############################
mem_pool_num=$(head -n 1 ${learner_iplist} | awk '{print $4}')
mem_pool_list='[35200'
for((i=1;i<${mem_pool_num};i++));
do
    mem_pool_port=$[35200+i]
    mem_pool_list=$mem_pool_list','$mem_pool_port
done
mem_pool_list=${mem_pool_list}']'
sed -i "s/ports =.*/ports = ${mem_pool_list}/g"  $LEARNER_DIR/config/common.conf
log "mem_pool_list: ${mem_pool_list}"


function run_learner(){
    log "start learner"
    cd ${LEARNER_DIR}
    export PATH=/data/opt/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/opt/ibutils/bin:/root/bin:$PATH
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
    nohup mpirun --allow-run-as-root -np ${gpu_num} -H ${node_list} -bind-to none -map-by slot \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_SOCKET_IFNAME=$NET_CARD_NAME \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH  python3 train.py >> ${LOG_DIR}/train.log 2>&1 &
}

##############################
if [[ $(cat ${learner_iplist} | wc -l) -ne 1 ]]
then
    # config sshd
    # 修改密码并启动sshd 
    rm -rf /root/.ssh/authorized_keys && ssh-keygen -t rsa -N '' -f /root/.ssh/id_rsa <<< y
    NEW_PASSWORD=${NEW_PASSWORD-${TASK_UUID-'random_pass_uuid'}}
    echo "root:${NEW_PASSWORD}" | chpasswd

    mkdir -p /run/sshd
    /usr/sbin/sshd

    # 等待sshd
    while true
    do
        continue_test=0
        while read line; 
        do
            line=$(echo $line | awk '{print $1}')
            echo "test $line"
            echo -e '\035' | telnet $line 36001 || continue_test=1
        done < ${SCRIPT_DIR}/gpu.iplist

        if [[ $continue_test -eq 0 ]]
        then
            echo "all test done"
            break
        fi
        sleep 1
    done

    cat ${learner_iplist} | awk '{print $1}'|while read line; 
    do
        ./ssh-copy-id.expect $line $NEW_PASSWORD
    done

    RANK_0=$(cat ${SCRIPT_DIR}/gpu.iplist|head -n 1| awk '{print $1}')
    if [[ "$(hostname -I| awk '{print $1}')" = "${RANK_0}" ]]
    then
        run_learner
    fi

else
    run_learner
fi

cd /rl_framework/send_model
nohup bash start_check_and_send_checkpoint.sh >> $LOG_DIR/send.log &
