#!/bin/bash

LOG_DIR=/aiarena/logs/
mkdir -p $LOG_DIR

if [ -n "$DEPLOY_GAMECORE" ]; then
    nohup sh /aiarena/remote-gc-server/run_and_monitor_gamecore_server.sh > $LOG_DIR/gamecore_server.log 2>&1 &
    GAMECORE_SERVER_ADDR="127.0.0.1:23432"
fi
GAMECORE_SERVER_ADDR=${GAMECORE_SERVER_ADDR-"127.0.0.1:23432"}


SCRIPT_DIR=$(realpath $(dirname $0))
cd $SCRIPT_DIR/../

SERVER_DRIVER_0=${SERVER_DRIVER_0-"url"}
SERVER_DRIVER_1=${SERVER_DRIVER_1-"url"}
WAIT_PORT_TIMEOUT=${WAIT_PORT_TIMEOUT-"30"}
SERVER_PORT_0=${SERVER_PORT_0-"35350"}
SERVER_PORT_1=${SERVER_PORT_1-"35351"}

python3 battle.py --server_0=$1 --server_1=$2 --server_driver_0=${SERVER_DRIVER_0} --server_driver_1=${SERVER_DRIVER_1} --wait_port_timeout=${WAIT_PORT_TIMEOUT} --server_port_0 ${SERVER_PORT_0} --server_port_1 ${SERVER_PORT_1} --gamecore_server=${GAMECORE_SERVER_ADDR}
