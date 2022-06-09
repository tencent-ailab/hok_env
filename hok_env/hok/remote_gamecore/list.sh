#!/bin/bash
# token, config
cd `dirname $0`
python remote_launcher.py list $1 "$2"