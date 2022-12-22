GAMECORE_PATH=${GAMECORE_PATH:-"/rl_framework/gamecore/"}
GAMECORE_SERVER_BIND_ADDR=${GAMECORE_SERVER_BIND_ADDR:-":23432"}

cd $GAMECORE_PATH

if [ -f "gamecore-server" ] && [ -z "${GAMECORE_SERVER_USE_WINE}" ]; then
    ./gamecore-server server --server-address=${GAMECORE_SERVER_BIND_ADDR}
else
    export WINEPATH="${GAMECORE_PATH}/lib/;${GAMECORE_PATH}/bin/"
    wine gamecore-server.exe server --server-address=${GAMECORE_SERVER_BIND_ADDR}
fi
