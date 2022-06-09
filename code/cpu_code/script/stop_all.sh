#!/bin/bash
cd actor_platform/; bash kill.sh
ps -ef | grep "./modelpool" | awk '{print $2}' | xargs kill -9
