ps -ef | grep "check_and_send_checkpoint.py" | awk '{print $2}' | xargs kill -9
