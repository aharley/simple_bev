#!/bin/bash

DATA_DIR="/mnt/fsx/nuscenes"
# DATA_DIR="../nuscenes/full_v1.0"
# DATA_DIR="../nuscenes"

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
# < ah but they just had one giant bin, that's why
EXP_NAME=lif15 # fewer prints
EXP_NAME=lif16 # 10k like this
EXP_NAME=lif17 # 100/10, where all Y coords are zero
EXP_NAME=lif18 # 4gpu, acc1, 10k like this
EXP_NAME=lif19 # 1gpu acc4
EXP_NAME=lif20 # 4gpu acc1, dset=trainval, 25k


EXP_NAME=lif24 # 4gpu acc1, dset=trainval, 25k < i don't have trainval
EXP_NAME=lif25 # mini, 100,10
# what is the problem?
# why are we getting overfitting, assuming the rest is correct?
EXP_NAME=lif26 # random depth
EXP_NAME=lif27 # 0.9 of hypoteneuse
EXP_NAME=lif28 # pred
EXP_NAME=lif21 # don't divide < smoother
EXP_NAME=lif22 # show me the middle slice
EXP_NAME=lif23 # and beg and end
EXP_NAME=lif24 # hm... middle slice again
EXP_NAME=lif25 # unproj ones
EXP_NAME=lif26 # divide by ones
EXP_NAME=lif27 # three ones
# why isn't the camera setup looking more consistent? 
EXP_NAME=lif28 # no crop
EXP_NAME=lif29 # don't shuffle cams
# ok looks solid
EXP_NAME=lif30 # 0.8 hyp
EXP_NAME=lif31 # show me the actual feat_bev
EXP_NAME=lif32 # 0.7
EXP_NAME=lif33 # 0.9 again
EXP_NAME=lif34 # rand
EXP_NAME=lif35 # softmax
EXP_NAME=lif36 # pred
EXP_NAME=lif37 # don't set y to 0
EXP_NAME=lif38 # feat=ones 
EXP_NAME=lif39 # 8 instead of 4 < it's already 8
EXP_NAME=lif40 # show me max
EXP_NAME=lif41 # pred
EXP_NAME=lif42 # show me mean again
EXP_NAME=lif43 # 1k
EXP_NAME=lif44 # ones, 100,10, better averaging
# feat_bev (float32) min = -12.72, mean = 0.00, max = 13.82
EXP_NAME=lif45 # ones, beg middle end
# min = 0.00, mean = 0.01, max = 1.00
# but not lining up
EXP_NAME=lif46 # don't clone
# still broken
EXP_NAME=lif47 # don't div by ones
EXP_NAME=lif48 # set y to 0
EXP_NAME=lif49 # set y to 1
# ok interesting:
# when i allow y to spread out, it goes everywhere
EXP_NAME=lif50 # don't set y
EXP_NAME=lif51 # ZYX when creating inds
EXP_NAME=lif52 # YX, X for coords
EXP_NAME=lif53 # div by output_ones
EXP_NAME=lif54 # pred
# let's compare cleanly
EXP_NAME=lif55 # don't print stats; go 1k
EXP_NAME=lif56 # don't div by ones, just to se
EXP_NAME=lif57 # div x_b also
EXP_NAME=lif58 # aws


# python train_nuscenes_bevseg.py \
#        --exp_name=${EXP_NAME} \
#        --max_iters=1000 \
#        --log_freq=100 \
#        --shuffle=True \
#        --dset='mini' \
#        --do_val=True \
#        --val_freq=100 \
#        --save_freq=9999999 \
#        --batch_size=1 \
#        --grad_acc=1 \
#        --lr=5e-4 \
#        --use_scheduler=True \
#        --weight_decay=1e-6 \
#        --nworkers=12 \
#        --data_dir=$DATA_DIR \
#        --log_dir='logs_nuscenes_bevseg' \
#        --ckpt_dir='checkpoints/' \
#        --do_rgbcompress=False \
#        --res_scale=1 \
#        --rand_flip=False \
#        --rand_crop_and_resize=False \
#        --do_shuffle_cams=False \
#        --ncams=6 \
#        --nsweeps=3 \
#        --encoder_type='res101' \
#        --device_ids=[0]

python train_nuscenes_bevseg.py \
       --exp_name=${EXP_NAME} \
       --max_iters=25000 \
       --log_freq=1000 \
       --shuffle=True \
       --dset='trainval' \
       --do_val=True \
       --val_freq=100 \
       --save_freq=1000 \
       --batch_size=8 \
       --grad_acc=5 \
       --lr=5e-4 \
       --use_scheduler=True \
       --weight_decay=1e-6 \
       --nworkers=12 \
       --data_dir=$DATA_DIR \
       --log_dir='/mnt/fsx1/bev_baseline/logs_nuscenes_bevseg' \
       --ckpt_dir='/mnt/fsx1/bev_baseline/checkpoints/' \
       --do_rgbcompress=False \
       --res_scale=2 \
       --rand_flip=True \
       --rand_crop_and_resize=True \
       --do_shuffle_cams=True \
       --ncams=6 \
       --nsweeps=3 \
       --encoder_type='res101' \
       --device_ids=[0,1,2,3,4,5,6,7]
