#!/bin/bash
# token, config
cd `dirname $0`
python remote_launcher.py check $1 "$2"