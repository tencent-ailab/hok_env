#!/bin/bash

ps -ef | grep "train.py" | awk '{print $2}' | xargs kill -9

process1=`ps -ef | grep "train.py" | grep -v grep | wc -l`

if [ $process1 -eq 0 ];then
   exit 0
else
   exit -1
fi
