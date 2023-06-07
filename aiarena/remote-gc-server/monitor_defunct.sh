while true
do
    ps -e f|grep -v monitor_defunct.sh|grep defunct|grep -v grep| awk '{print $1}'|xargs kill -s 9
    sleep 10
done
