#!/bin/bash

function del_file(){
    file=$1
    if [ -f "$file" ]; then
        rm "$file"
    fi
}

function del_dir(){
    file=$1
    if [ ! -d $file ]; then
        mkdir $file
    else
        rm -r $file && mkdir $file
    fi
}
       
del_file "start.log"
del_dir "log"
del_dir "model/update"
del_dir "../send_model/model"
rm -rf log; mkdir log
