mkdir -p /aiarena/code/actor/log/
python entry.py --actor_id=0 \
    --mem_pool_addr=localhost:35200 \
    --model_pool_addr=localhost:10016 \
    --gc_server_addr=${GAMECORE_SERVER_ADDR-"127.0.0.1:23432"} \
    --ai_server_ip=${AI_SERVER_IP-`hostname -I | awk '{print $1;}'`} \
    --thread_num=1 \
    --single_test
