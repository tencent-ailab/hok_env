#!/bin/bash

ps -e f|grep -E "monitor_actor|influxdb|python|modelpool"| awk '{print $1}'|xargs kill -s 9

