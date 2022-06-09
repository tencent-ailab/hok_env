#!/bin/bash
# token, path, config -> token, config, path
cd `dirname $0`
python remote_launcher.py get_path $1 "$3" $2
