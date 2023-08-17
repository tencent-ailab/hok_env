GAMECORE_PATH=${GAMECORE_PATH:-"/rl_framework/gamecore/"}
GAMECORE_SERVER_BIND_ADDR=${GAMECORE_SERVER_BIND_ADDR:-":23432"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $GAMECORE_PATH

if [ "${REPORTER_MONGODB_URI}" != "" ]; then
    args+=("--reporter-callback-enable" "--reporter-mongodb-uri=${REPORTER_MONGODB_URI}")
fi

if [ -f "gamecore-server-linux-amd64" ] && [ -z "${GAMECORE_SERVER_USE_WINE}" ]; then
    if [ -z "${SIMULATOR_USE_WINE}" ]; then
        ./gamecore-server-linux-amd64 server --server-address=${GAMECORE_SERVER_BIND_ADDR} "${args[@]}"
    else
        ./gamecore-server-linux-amd64 server --server-address=${GAMECORE_SERVER_BIND_ADDR} \
            --simulator-remote-bin ${SCRIPT_DIR}/sgame_simulator_remote_zmq \
            --simulator-repeat-bin ${SCRIPT_DIR}/sgame_simulator_repeated_zmq \
            "${args[@]}"
    fi
else
    export WINEPATH="${GAMECORE_PATH}/lib/;${GAMECORE_PATH}/bin/"
    wine gamecore-server.exe server --server-address=${GAMECORE_SERVER_BIND_ADDR} "${args[@]}"
fi
