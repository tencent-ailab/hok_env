#!/bin/bash
# sh script/setup_parameters.sh task/$task_name/gpu.iplist task/$task_name/cpu.iplist
#$training_type $steps $env_name $task_name $action_dim
rm send.log
if [ $# -lt 5 ];then
echo "usage $0 gpu_iplist cpu_iplist mem_pool_num task_id task_uuid"
exit
fi

gpu_iplist=$1
cpu_iplist=$2
mem_pool_num=$3
task_id=$4
task_uuid=$5

docker_id=`hostname |awk -F '-' '{print $NF}'`
echo "`hostname -i` root 36001 $CPU_NUM" > /code/code/cpu_code/script/cpu.iplist

#get gpu mempool_list
cd /code/code/cpu_code/script/
gpu_ips_num=`cat $gpu_iplist | awk '{print $1}' | wc -l`
cpu_ips_num=`cat $cpu_iplist | awk '{print $1}' | wc -l`

echo "task_id=$task_id" > task.cfg
echo "task_uuid=${task_uuid}" >> task.cfg

rm mem_pool.host_list
let end_num=mem_pool_num-1
echo "end_num",$end_num
while read ip user port gpu_card_num;do
    echo $ip
    for i in `seq 0 $end_num`
    do
        let port=35200+$i
        echo "$ip:$port"
        echo "$ip:$port" >> mem_pool.host_list
    done


done < $gpu_iplist
cp mem_pool.host_list mem_pool.host_list.all
python set_current_ip_2_mem_pool.py $cpu_iplist mem_pool.host_list $docker_id > current.iplist

mv mem_pool.host_list /code/code/cpu_code/actor/config/mem_pool.host_list

