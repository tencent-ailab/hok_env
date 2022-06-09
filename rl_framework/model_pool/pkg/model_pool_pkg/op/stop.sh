#!/bin/bash
rm -rf ../bin/files
rm -rf ../bin/model

ps -ef | grep "modelpool" | awk '{print $2}' | xargs kill -9

process1=`ps -ef | grep "modelpool" | grep -v grep | wc -l`

if [ $process1 -eq 0 ];then
   exit 0
else
   exit -1
fi
