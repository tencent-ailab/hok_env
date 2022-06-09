#!/bin/bash

while :
do
    #rm ./update/*
    len=`ls ./model/update | wc -l`
    len1=`expr $len - 200`
    echo $len 
    echo $len1
    if [ $len -gt 220 ]
    then
        cd ./model/update
        ls -rt | head -n $len1 | xargs -I {} rm {}
        cd -
    fi
    sleep 600
done
