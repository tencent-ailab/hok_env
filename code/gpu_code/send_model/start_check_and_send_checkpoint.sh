#!/bin/bash
if [ -d "model" ]; then
    rm -r ./model/ && mkdir model
fi


mkdir /model_bkup

# start send model
cd code &&
nohup python check_and_send_checkpoint.py \
    --predictor_type=local \
    --syn_type=model_pool \
    --address=127.0.0.1:10013:10014 \
    --is_delete=True > ../check_and_send.log 2>&1 &
