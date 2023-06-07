#!/bin/bash

ps -e f|grep -E "python|modelpool"| awk '{print $1}'|xargs kill -s 9

