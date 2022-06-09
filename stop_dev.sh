ps -ef | grep "sgame_simulator_" | awk '{print $2}' | xargs kill -9
sleep 1s
ps -ef | grep "entry" | awk '{print $2}' | xargs kill -9
sleep 1s
ps -ef | grep "train" | awk '{print $2}' | xargs kill -9
sleep 1s
ps -ef | grep "model" | awk '{print $2}' | xargs kill -9
sleep 1s
ps -ef | grep "check" | awk '{print $2}' | xargs kill -9
