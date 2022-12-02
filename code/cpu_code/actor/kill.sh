ps -aux | grep sgame_simulator_ | grep -v grep |awk '{print $2}' | xargs kill -9 2>/dev/null
ps -aux | grep entry.py | grep -v grep |awk '{print $2}' | xargs kill -9 2>/dev/null
