#/bin/bash

gpu_iplist=./gpu.iplist
cpu_iplist=./cpu.iplist

. ./config.conf
if [ $mem_pool_num ];then
    echo "mempool_num:", $mem_pool_num
else
    mem_pool_num=8
fi
if [ $steps ];then
    echo "steps:", $steps
else
    steps=128
fi
if [ $use_zmq ];then
    echo "use_zmq:", $use_zmq
    if [ $use_socket ];then
        echo "use_socket:", $use_socket
    else
        use_socket=0
    fi
else
    if [ $use_socket ];then
        echo "use_socket:", $use_socket
        use_zmq=0
    else
        use_zmq=0
        use_socket=1
    fi
fi

type1="async"
type2="sync"
echo "howe start set_gpu"${use_zmq}

ips_list=`cat $gpu_iplist |awk '{print $1}'`
ips_array=($(cat $gpu_iplist |awk '{print $1}'))
ips_num=`cat $gpu_iplist | awk '{print $1}' | wc -l`
ips=""
i=0

for ip in $ips_list;do
if [ $i -eq 0 ];then
    ips="$ip"
else
    ips="$ips,$ip"
fi
let i+=1
done

gpu_num_list=`cat $gpu_iplist | awk '{print $4}'`
gpu_num=`cat $gpu_iplist | awk '{print $4}' | awk '{sum+=$1} END {print sum}'`
actor_num=`cat $cpu_iplist | awk '{print $4}' | awk '{sum+=$1} END {print sum}'`

echo gpu_num, $gpu_num
echo actor_num, $actor_num
echo gpu_ips, $ips

node_list=""
gpu_list=\"
gpu_idx=0
init_port=35910
port=$init_port
for tmp in $gpu_num_list
do
    gpu_ip=${ips_array[$gpu_idx]}
    if [ $gpu_idx -eq 0 ]
    then
        node_list="$gpu_ip:$tmp"
    else
        node_list="$node_list,$gpu_ip:$tmp"
    fi
    let gpu_idx+=1

    for i in $(seq 1 $tmp)
    do
        let "port+=1"
        if [[ "$i" -ne "$tmp" ]] || [[ "$gpu_idx" -ne "$ips_num" ]]
        then
	    gpu_list=$gpu_list$gpu_ip\:$port\;
        else
	    gpu_list=$gpu_list$gpu_ip\:$port
        fi
    done
done

gpu_list=$gpu_list\"
echo gpu_list, $gpu_list
echo Nodelist, $node_list

if [[ $use_socket == 1 ]];then
   sed -i 's/network_dataset_dataop/network_dataset_socket_async/g' /code/code/gpu_code/learner/code/train.py
fi

if [[ $use_zmq == 1 ]];then
   sed -i 's/network_dataset_dataop/network_dataset_zmq_dataset/g' /code/code/gpu_code/learner/code/train.py
fi

sed -i "s/ips =.*/ips = ${ips}/g" /code/code/gpu_code/learner/code/common.conf
sed -i "s/training_type =.*/training_type = $training_type/g" /code/code/gpu_code/learner/code/common.conf

if [ "${training_type}"x == "${type1}"x ]
then
    mem_pool_list='[35200'
    for((i=1;i<${mem_pool_num};i++));
    do
        mem_pool_port=$[35200+i]
        mem_pool_list=$mem_pool_list','$mem_pool_port
    done
    mem_pool_list=${mem_pool_list}']'
    sed -i "s/ports =.*/ports = ${mem_pool_list}/g"  /code/code/gpu_code/learner/code/common.conf
    #send model cpu_iplist
    rm /framework/send_model/mcp_opporation_tools/current.iplist
    count=0
    while read ip user port num;do
        count=$[count+1]
        mod=$[count%5]
        echo $ip $user $port $num
        echo $ip $port >> /framework/send_model/mcp_opporation_tools/current.iplist
        if [ $mod -eq 0 ];then
            echo $mod, $count
            wait
        fi
    done <$cpu_iplist
    wait
fi
sed -i "s/Num_process=.*/Num_process=$gpu_num/g" /code/code/gpu_code/learner/code/run_multi.sh
sed -i "s/Nodelist=.*/Nodelist=$node_list/g" /code/code/gpu_code/learner/code/run_multi.sh

