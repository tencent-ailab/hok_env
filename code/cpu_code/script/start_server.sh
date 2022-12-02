
SCRIPT_DIR=$(realpath $(dirname $0))
cd $SCRIPT_DIR/../actor/code/

AI_SERVER_ADDR=${AI_SERVER_ADDR-"tcp://0.0.0.0:35400"}
python3 ./server.py --model_path="$SCRIPT_DIR/../actor/code/algorithms/checkpoint" --server_addr=${AI_SERVER_ADDR}
