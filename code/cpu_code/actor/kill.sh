if [ ! $GAMECORE_SERVER_ADDR ]; then
  curl -k https://127.0.0.1:23432/v1/stopGame -d '{"Token": "127D0D0D1","CustomConfig": "{\"runtime_id\":-1}"}'
else
  TOKEN=$(echo ${AI_SERVER_ADDR} | awk '{ gsub(/\./,"D"); print $0 }')
  echo "Kill all running processes of token:${TOKEN}."
  # runtime_id = -1 means to kill all running processes.
  curl -k https://${GAMECORE_SERVER_ADDR}/v1/stopGame -d '{"Token": "'"${TOKEN}"'","CustomConfig": "{\"runtime_id\":-1}"}'
fi

echo "Seed [KILL_ALL] Command to remote game server, please wait for 1 sec."
sleep 1

ps -aux |grep sgame_simulator_ |grep -v grep |awk '{print $2}' |xargs kill -9
ps -aux |grep entry.py |grep -v grep |awk '{print $2}' |xargs kill -9
