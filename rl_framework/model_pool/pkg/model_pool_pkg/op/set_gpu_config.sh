#!/bin/bash

ip=`hostname -i`

cd ../config && rm trpc_go.yaml
cd ../config && cp trpc_go.yaml.gpu trpc_go.yaml

sed -i "s/__MODELPOOL_IP_HERE__/${ip}/g" ../config/trpc_go.yaml
