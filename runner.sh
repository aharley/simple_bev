#!/bin/bash

DATA_DIR="/mnt/fsx/nuscenes"
# there should be ${DATA_DIR}/trainval/v1.0-trainval

# rgb00: default, on gpus 0-3
# rgb01: flip=True, on gpus 4-7

python nuscenes_bevseg.py \
       --exp_name="rgb01" \
       --max_iters=20000 \
       --log_freq=1000 \
       --shuffle=True \
       --dset='trainval' \
       --do_val=True \
       --val_freq=100 \
       --save_freq=1000 \
       --batch_size=40 \
       --batch_parallel=8 \
       --lr=3e-4 \
       --weight_decay=1e-7 \
       --grad_acc=1 \
       --nworkers=12 \
       --data_dir=$DATA_DIR \
       --log_dir='logs_nuscenes_bevseg' \
       --ckpt_dir='checkpoints/' \
       --keep_latest=3 \
       --init_dir='' \
       --ignore_load='' \
       --load_step=False \
       --load_optimizer=False \
       --resolution_scale=2 \
       --rand_flip=True \
       --ncams=6 \
       --nsweeps=3 \
       --encoder_type='res101' \
       --use_radar=False \
       --use_lidar=False \
       --do_metaradar=False \
       --do_rgbcompress=True \
       --device='cuda:4' \
       --device_ids=[4,5,6,7] \
       # --device_ids=[0,1,2,3] \


