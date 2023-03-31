#!/bin/bash

DATA_DIR="../nuscenes"
DATA_DIR="/data2/nuscenes"
# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

MODEL_NAME="1x5_3e-4_rgb00_20:04:28"


# python eval_nuscenes.py \
#        --batch_size=1 \
#        --exp_name=${EXP_NAME} \
#        --dset='mini' \
#        --data_dir=$DATA_DIR \
#        --log_dir='logs_eval_nuscenes' \
#        --init_dir="checkpoints/${MODEL_NAME}" \
#        --res_scale=1 \
#        --device_ids=[0]


MODEL_NAME="8x5_3e-4s_rrl00_20:08:53"
EXP_NAME="00" # evaluate rrl00 model
# final trainval mean iou 64.82080960497514
# insane

MODEL_NAME="8x5_3e-4s_rrl01_19:17:52"
EXP_NAME="01" # evaluate rrl01 model (eval3 segnet3), which uses just metarad, but with the alternate squash strategy
# 55.02691704140531

MODEL_NAME="8x5_3e-4s_rrl02_19:22:47"
EXP_NAME="02" # evaluate rrl02 model (eval4 segnet4), which uses just lidar
# 63.71491550846539
# ok better than the paper
# but i think that makes sense, that was an old value

python eval_nuscenes4.py \
       --batch_size=4 \
       --exp_name=${EXP_NAME} \
       --dset='trainval' \
       --data_dir=$DATA_DIR \
       --log_dir='logs_eval_nuscenes' \
       --init_dir="checkpoints/${MODEL_NAME}" \
       --res_scale=2 \
       --ncams=6 \
       --encoder_type='res101' \
       --do_rgbcompress=True \
       --use_radar=True \
       --use_lidar=True \
       --use_metaradar=True \
       --device_ids=[0,1] 
