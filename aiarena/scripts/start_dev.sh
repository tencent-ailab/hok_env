function log(){
  now=`date +"%Y-%m-%d %H:%M:%S"`
  echo "[$now] $1"
}
export NO_ACTOR_MODEL_POOL=1

LOG_DIR=/aiarena/logs/
mkdir -p $LOG_DIR

log "---------------------------starting learner-----------------------------"
cd /aiarena/scripts/learner
nohup sh start_learner.sh >> $LOG_DIR/start_learner.log 2>&1 &

log "---------------------------starting actor-----------------------------"
cd /aiarena/scripts/actor
nohup sh start_actor.sh >> $LOG_DIR/start_actor.log 2>&1 &
