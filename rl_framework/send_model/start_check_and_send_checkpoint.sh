#!/bin/bash

if [ -d "model" ]; then
    rm -r ./model/ && mkdir model
fi

enable_backup=False
if [ -n "$ENABLE_SEND_MODEL_BACKUP" ]; then
    echo "enable backup dir"
    enable_backup=True
fi

cd code &&
python3 check_and_send_checkpoint.py \
    --predictor_type=local \
    --syn_type=model_pool \
    --address=127.0.0.1:10013:10014 \
    --is_delete=True \
    --enable_backup=$enable_backup
