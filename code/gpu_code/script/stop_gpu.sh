#!/bin/bash
# Stop RL learners and Memory pools.
export USER=root
. ./config.conf
end_port=$[35200 + $mem_pool_num - 1] 
cd /code/code/gpu_code/learner/ && bash kill.sh

