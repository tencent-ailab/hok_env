#!/bin/bash

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [<file>] [<max_time_diff>]"
    exit 1
fi

file=${1-${CHECK_FILE_UPDATE_FILE-"/aiarena/logs/learner/train.log"}}
max_time_diff=${2-${CHECK_FILE_UPDATE_MAX_TIME_DIFF-"300"}}

if [ ! -e "$file" ]; then
    echo "File not found: $file"
    exit 2
fi

current_time=$(date +%s)
file_mod_time=$(stat -c %Y "$file")
time_diff=$((current_time - file_mod_time))

if [ $time_diff -gt ${max_time_diff} ]; then
    echo "The file ${file} has not been modified in the last ${max_time_diff} s."
    exit 3
fi
