#!/bin/bash

LOG_DIR=/aiarena/logs/
mkdir -p $LOG_DIR

nohup sh /rl_framework/remote-gc-server/run_and_monitor_gamecore_server.sh >/dev/null 2>&1 &
while true; do
    lsof -i ${GAMECORE_SERVER_BIND_ADDR} && break
    sleep 1
done
