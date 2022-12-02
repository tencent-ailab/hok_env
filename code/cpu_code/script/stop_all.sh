#!/bin/bash
cd ../actor/; bash kill.sh
ps -ef | grep "./modelpool" | awk '{print $2}' | xargs kill -9
