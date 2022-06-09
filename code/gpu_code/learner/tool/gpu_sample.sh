#!/usr/bin/bash

getdata=0
is_get=`cat /data1/reinforcement_platform/rl_learner_platform/log/train.log | grep 'images/sec mean' | wc -l`
if [ $is_get -gt 0 ];then
   getdata=`cat /data1/reinforcement_platform/rl_learner_platform/log/train.log | grep 'images/sec mean' | tail -n 10 | awk -F 'images/sec mean' '{print $2}' | awk '{sum+=$2} END {print sum/NR}' | awk '$1=$1' | cut -d '.' -f1`
   output=`nvidia-smi 2>&1`
   error='command not found'
   if [[ $output =~ $error  ]];then
       getdata=0
   else
       gpu_num=`nvidia-smi | grep N/A | wc -l`
       getdata=$[getdata*60*gpu_num] 
   fi
fi

setdata=0
is_dataset=`cat /data1/reinforcement_platform/rl_learner_platform/log/train.log | grep recv_sample/sec | wc -l`
if [ $is_dataset -gt 0 ];then
   setdata=`cat /data1/reinforcement_platform/rl_learner_platform/log/train.log | grep recv_sample/sec | tail -n 10 | awk -F 'recv_sample/sec' '{print $2}' | awk '{sum+=$2} END {print sum/NR}' | awk '$1=$1' | cut -d '.' -f1`
   output=`nvidia-smi 2>&1`
   error='command not found'
   if [[ $output =~ $error  ]];then
       echo 0,0
   else
       gpu_num=`nvidia-smi | grep N/A | wc -l`
       setdata=$[setdata*60*gpu_num] 
       echo $setdata,$getdata
   fi
   exit 0
fi

files=`ls /data1/reinforcement_platform/mem_pool_server_pkg| grep mem_pool_server | grep -v mem_pool_server_pkg`
setdata=0
for file in $files
do
    cd "/data1/reinforcement_platform/mem_pool_server_pkg/$file/log"
    log=`ls | grep stat.log`
    if [ $log ];then
        temp=`cat ${log} | grep OnReqSetData | tail -n 1 | awk -F '|' '{print $2}' | awk '$1=$1' `
        if [[ $temp != *[!0-9]* ]];then
            setdata=`expr $setdata + $temp`
        fi
    fi
done
echo "generation_rate/min:"$setdata
echo "consumption_rate/min:"$getdata
echo $setdata,$getdata
