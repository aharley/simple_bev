#!/bin/bash

DATA_DIR="/data2/nuscenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

EXP_NAME="rgb00" # default settings
EXP_NAME="rgb01" # removed some code duplication
EXP_NAME="rgb02" # cleaned up dataset file
EXP_NAME="rgb03" # updated log dir
EXP_NAME="rgb04" # go feb23; match settings 
EXP_NAME="rgb05" # mini
EXP_NAME="rgb06" # try bevformer
EXP_NAME="rrl00" # train2 segnet2 with radar AND lidar

EXP_NAME="rrl01" # train3 segnet3 with just metarad on gpu0,1; note that here we do NOT sum across the vert 
EXP_NAME="rrl02" # train4 segnet4 with just lidar on gpu1,2

# python train_nuscenes2.py \
#        --exp_name=${EXP_NAME} \
#        --max_iters=25000 \
#        --log_freq=1000 \
#        --dset='mini' \
#        --batch_size=1 \
#        --grad_acc=1 \
#        --use_scheduler=True \
#        --data_dir=$DATA_DIR \
#        --log_dir='./logs_nuscenes' \
#        --ckpt_dir='./checkpoints' \
#        --res_scale=2 \
#        --ncams=6 \
#        --encoder_type='res101' \
#        --do_rgbcompress=True \
#        --use_radar=True \
#        --use_lidar=True \
#        --use_metaradar=True \
#        --device_ids=[0] 

python train_nuscenes4.py \
       --exp_name=${EXP_NAME} \
       --max_iters=25000 \
       --log_freq=1000 \
       --dset='trainval' \
       --batch_size=8 \
       --grad_acc=5 \
       --use_scheduler=True \
       --data_dir=$DATA_DIR \
       --log_dir='./logs_nuscenes' \
       --ckpt_dir='./checkpoints' \
       --res_scale=2 \
       --ncams=6 \
       --encoder_type='res101' \
       --do_rgbcompress=True \
       --use_radar=True \
       --use_lidar=True \
       --use_metaradar=True \
       --device_ids=[0,1] 


# python train_nuscenes.py \
#        --exp_name=${EXP_NAME} \
#        --max_iters=25000 \
#        --log_freq=1000 \
#        # --dset='trainval' \
#        --dset='mini' \
#        --batch_size=8 \
#        --grad_acc=5 \
#        --use_scheduler=True \
#        --data_dir=$DATA_DIR \
#        --log_dir='./logs_nuscenes' \
#        --ckpt_dir='./checkpoints' \
#        --res_scale=2 \
#        --ncams=6 \
#        --encoder_type='res101' \
#        --do_rgbcompress=True \
#        --device_ids=[0,1,2,3] 

