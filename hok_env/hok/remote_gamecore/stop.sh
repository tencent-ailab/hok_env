#!/bin/bash
# token, config
cd `dirname $0`
python remote_launcher.py stop $1 "$2"