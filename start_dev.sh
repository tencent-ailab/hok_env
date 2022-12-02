export GC_MODE="remote"
USE_GPU=true
if [ $use_gpu ];then
    USE_GPU=$use_gpu
fi
if [ $USE_GPU == false ] && [ `pip list |grep -c tensorflow ` -eq 2 ];then
    cd /;
    pip install tensorflow-1.14.0-cp36-cp36m-manylinux1_x86_64.whl
fi


rm -rf /code/logs
mkdir -p /code/logs

echo "---------------------------starting gpu-----------------------------"
cd /code/code/gpu_code/script
nohup sh start_gpu.sh >/code/logs/start_gpu.log 2>&1 &

echo "---------------------------starting cpu-----------------------------"
cd /code/code/cpu_code/script
nohup sh start_cpu.sh >/code/logs/start_cpu.log 2>&1 &
