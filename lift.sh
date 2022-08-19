#!/bin/bash

DATA_DIR="../nuscenes"

# lif00: debug
# lif01: show me feat_bev
# lif02: ncams=6, batch1, grad8
# lif03: ncams=6, batch1, grad1
# lif04: 1k iters
# lif05: 100, liftnet
EXP_NAME=lif06 # uniform dist
EXP_NAME=lif07 # 4 to Z
EXP_NAME=lif08 # 4 to Z+4
EXP_NAME=lif09 # 5 to Z+5
EXP_NAME=lif10 # div by ones
EXP_NAME=lif11 # non-uniform
EXP_NAME=lif12 # /0.07
EXP_NAME=lif13 # train 10k
# ok, the fiery code is quite good... let's try that. once i get coffee anyway.
EXP_NAME=lif14 # fiery 100
# only weird thing is that i'm forced to sum over bev myself
EXP_NAME=lif15 # fewer prints
EXP_NAME=lif16 # 10k like this
EXP_NAME=lif17 # 100/10, where all Y coords are zero
EXP_NAME=lif18 # 4gpu, acc1, 10k like this
EXP_NAME=lif19 # 1gpu acc4
EXP_NAME=lif20 # 4gpu acc1, dset=trainval, 25k

python train_nuscenes_bevseg.py \
       --exp_name=${EXP_NAME} \
       --max_iters=25000 \
       --log_freq=1000 \
       --shuffle=True \
       --dset='trainval' \
       --do_val=True \
       --val_freq=100 \
       --save_freq=9999999 \
       --batch_size=4 \
       --grad_acc=1 \
       --lr=5e-4 \
       --use_scheduler=True \
       --weight_decay=1e-6 \
       --nworkers=12 \
       --data_dir=$DATA_DIR \
       --log_dir='logs_nuscenes_bevseg' \
       --ckpt_dir='checkpoints/' \
       --do_rgbcompress=False \
       --res_scale=1 \
       --rand_flip=False \
       --ncams=6 \
       --nsweeps=3 \
       --encoder_type='res101' \
       --device_ids=[0,1,2,3]
