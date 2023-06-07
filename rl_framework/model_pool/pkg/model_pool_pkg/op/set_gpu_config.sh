#!/bin/bash

ip=`hostname -I | awk '{print $1;}'`

cd ../config && rm trpc_go.yaml
cd ../config && cp trpc_go.yaml.gpu trpc_go.yaml

sed -i "s/__MODELPOOL_IP_HERE__/${ip}/g" ../config/trpc_go.yaml
