ps -eo "%p %a" | sed '/bash$/d' | awk '{print $1}' | sed '/^1$/d'|grep -v PID|xargs kill -s 9
