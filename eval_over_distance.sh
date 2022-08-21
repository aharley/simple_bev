#!/bin/bash

DATA_DIR="../nuscenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

MODEL_NAME="1x5_3e-4_rgb00_20:04:28"
MODEL_NAME="8x5_5e-4_rgb12_22:43:46"

EXP_NAME="evd00" # evaluate 8x5_5e-4_rgb12_22:43:46 on mini just to go
EXP_NAME="evd01" # eval_over_distance.py; compute dist map
EXP_NAME="evd02" # 20 bins

python eval_over_distance.py \
       --batch_size=1 \
       --exp_name=${EXP_NAME} \
       --dset='mini' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_over_distance' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --res_scale=1 \
       --device_ids=[0]
