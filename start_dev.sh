echo "---------------------------starting gpu-----------------------------"
# create MODEL backup dir
if [ ! -d "/model_bkup" ]; then
    mkdir /model_bkup
fi
if [ ! -d "/logs/cpu_log" ]; then
    mkdir -p /logs/cpu_log
fi
if [ ! -d "/logs/gpu_log" ]; then
    mkdir -p /logs/gpu_log
fi
chmod -R +755 /code
cd /code/gpu_code/script

# start learner
sh start_gpu.sh
sleep 15s
echo "---------------------------starting cpu-----------------------------"
cd /code/cpu_code/script

# start actor
sh start_cpu.sh
sleep 5s
mkdir /code/logs
