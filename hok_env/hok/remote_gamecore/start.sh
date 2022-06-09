#!/bin/bash
# token, config
cd `dirname $0`
python remote_launcher.py start $1 "$2"
PY_RET=$?
find /root/remote-gc-server/log/ -mtime +1 -name "*.log" -exec rm -rfv {} \; &
return $PY_RET