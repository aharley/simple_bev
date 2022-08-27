#!/bin/bash

DATA_DIR="../lyft"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

EXP_NAME="rgb00" # swap nusc for lyft, and fix
EXP_NAME="rgb01" # Y range -10,10, shuf=False
EXP_NAME="rgb02" # down a meter
EXP_NAME="rgb03" # -5,5 again
EXP_NAME="rgb04" # shuffle=true
EXP_NAME="rgb05" # train

# python train_lyft.py \
#        --exp_name=${EXP_NAME} \
#        --max_iters=10 \
#        --log_freq=2 \
#        --dset='lyft' \
#        --do_shuffle_cams=False \
#        --batch_size=1 \
#        --grad_acc=1 \
#        --data_dir=$DATA_DIR \
#        --log_dir='logs_lyft' \
#        --ckpt_dir='checkpoints' \
#        --res_scale=2 \
#        --ncams=6 \
#        --nsweeps=1 \
#        --encoder_type='res101' \
#        --do_rgbcompress=True \
#        --device_ids=[0] 

python train_lyft.py \
       --exp_name=${EXP_NAME} \
       --max_iters=25000 \
       --log_freq=1000 \
       --dset='lyft' \
       --batch_size=16 \
       --grad_acc=3 \
       --val_freq=100 \
       --save_freq=1000 \
       --lr=5e-4 \
       --use_scheduler=True \
       --weight_decay=1e-6 \
       --nworkers=12 \
       --data_dir=$DATA_DIR \
       --log_dir='/mnt/fsx1/bev_baseline/logs_lyft' \
       --ckpt_dir='/mnt/fsx1/bev_baseline/checkpoints/' \
       --res_scale=2 \
       --ncams=6 \
       --nsweeps=1 \
       --encoder_type='res101' \
       --do_rgbcompress=True \
       --device_ids=[0,1,2,3,4,5,6,7] 

