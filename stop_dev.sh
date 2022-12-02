#!/bin/bash
cd /code/code/cpu_code/script; bash stop_all.sh
cd /code/code/gpu_code/script; bash stop_gpu.sh
ps aux | grep check_and_send_checkpoint | grep -v grep | awk '{print $2}' | xargs kill -9
ps aux | grep config | grep -v grep | awk '{print $2}' | xargs kill -9
ps aux | grep start_gpu.sh | grep -v grep | awk '{print $2}' | xargs kill -9
ps aux | grep backup_model_code.sh | grep -v grep | awk '{print $2}' | xargs kill -9
