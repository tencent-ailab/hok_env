#!/bin/bash

if [ $# -lt 1 ];then
echo "usage $0 master_ip"
exit -1
fi

#MODELPOOL_ADDR=$1
MODELPOOL_ADDR=$1":10013"

ip=`hostname -I | awk '{print $1;}'`
TVMEC_DOCKER_ID=`hostname`
if [ -z "$CLUSTER_CONTEXT" ];then
    CLUSTER_CONTEXT='default'
fi

cd ../config && rm trpc_go.yaml
cd ../config && cp trpc_go.yaml.cpu trpc_go.yaml

sed -i "s/__TARGET_TRPC_ADDRESS_HERE__/${MODELPOOL_ADDR}/g" ../config/trpc_go.yaml
sed -i "s/__MODELPOOL_CLUSTER_HERE__/${CLUSTER_CONTEXT}/g" ../config/trpc_go.yaml
sed -i "s/__MODELPOOL_IP_HERE__/${ip}/g" ../config/trpc_go.yaml
sed -i "s/__MODELPOOL_NAME_HERE__/${TVMEC_DOCKER_ID}/g" ../config/trpc_go.yaml
