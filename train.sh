#!/bin/bash

DATA_DIR="../nuscenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

EXP_NAME="rgb00" # default settings
EXP_NAME="rgb01" # removed some code duplication
EXP_NAME="rgb02" # cleaned up dataset file
EXP_NAME="rgb03" # updated log dir

python train_nuscenes.py \
       --exp_name=${EXP_NAME} \
       --max_iters=25000 \
       --log_freq=1000 \
       --dset='mini' \
       --batch_size=1 \
       --grad_acc=5 \
       --data_dir=$DATA_DIR \
       --log_dir='logs_nuscenes' \
       --ckpt_dir='checkpoints' \
       --res_scale=1 \
       --ncams=6 \
       --encoder_type='res101' \
       --do_rgbcompress=True \
       --device_ids=[0] 

