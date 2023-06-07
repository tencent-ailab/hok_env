ACTOR_CODE_DIR=${ACTOR_CODE_DIR:-"/aiarena/code/actor"}
cd $ACTOR_CODE_DIR

CHECKPOINT_DIR=${ACTOR_CODE_DIR}/model/init

AI_SERVER_ADDR=${AI_SERVER_ADDR-"tcp://0.0.0.0:35400"}
python3 ./server.py --model_path="${CHECKPOINT_DIR}" --server_addr=${AI_SERVER_ADDR}
