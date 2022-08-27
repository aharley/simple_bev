#!/bin/bash

DATA_DIR="../nuscenes"
DATA_DIR="/mnt/fsx/lyft"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

MODEL_NAME="16x3_5e-4s_rgb06_01:34:39"

EXP_NAME="eval" # eval 16x3_5e-4s_rgb06_01:34:39 lyft
EXP_NAME="eval" # 416, B16
EXP_NAME="eval" # 1920,1080 base resolution
EXP_NAME="eval" # move centroid down 1 < 24.03662218212872
EXP_NAME="eval" # 1024 instead of 1080 (?)

python eval_lyft.py \
       --exp_name=${EXP_NAME} \
       --batch_size=16 \
       --dset='lyft' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_lyft' \
       --init_dir="/mnt/fsx1/bev_baseline/checkpoints/${MODEL_NAME}" \
       --res_scale=2 \
       --device_ids=[0,1,2,3,4,5,6,7]
