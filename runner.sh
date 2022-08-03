#!/bin/bash

DATA_DIR="/mnt/fsx/nuscenes"
# there should be ${DATA_DIR}/trainval/v1.0-trainval

# rgb00: default, on gpus 0-3
# rgb01: flip=True, on gpus 4-7
# rgb02: resume rgb01, go to 25k, since resize_lim was .31 to .35
# rgb03: resume rgb01, go to 25k, since resize_lim was .31 to .35, lr 3e-5
# i do think that it would be better to have symmetric scaling around our golden 0.31
# and this means: i need to deal with cropping and scaling
# actually, resize_lim=None in these...
# python nuscenes_bevseg.py \
#        --exp_name="rgb03" \
#        --max_iters=25000 \
#        --log_freq=1000 \
#        --shuffle=True \
#        --dset='trainval' \
#        --do_val=True \
#        --val_freq=100 \
#        --save_freq=1000 \
#        --batch_size=40 \
#        --batch_parallel=8 \
#        --lr=3e-4 \
#        --weight_decay=1e-7 \
#        --grad_acc=1 \
#        --nworkers=12 \
#        --data_dir=$DATA_DIR \
#        --log_dir='logs_nuscenes_bevseg' \
#        --ckpt_dir='checkpoints/' \
#        --keep_latest=3 \
#        --init_dir='checkpoints/40_3e-4_rgb01_01:13:43' \
#        --ignore_load='' \
#        --load_step=True \
#        --load_optimizer=False \
#        --resolution_scale=2 \
#        --rand_flip=True \
#        --ncams=6 \
#        --nsweeps=3 \
#        --encoder_type='res101' \
#        --use_radar=False \
#        --use_lidar=False \
#        --do_metaradar=False \
#        --do_rgbcompress=True \
#        --device='cuda:0' \
#        --device_ids=[0,1,2,3] 
#        # --device='cuda:4' \
#        # --device_ids=[4,5,6,7] \



# rgb04: use new dataset file, with resize_lim=0.9-1.1 and res 224x400; go to 25k at 3e-4 < resume and go to 40k
# rgb05: same but also use linear scheduler
# python nuscenes_bevseg.py \
#        --exp_name="rgb06" \
#        --max_iters=25000 \
#        --log_freq=1000 \
#        --shuffle=True \
#        --dset='trainval' \
#        --do_val=True \
#        --val_freq=100 \
#        --save_freq=1000 \
#        --batch_size=40 \
#        --batch_parallel=8 \
#        --lr=3e-4 \
#        --weight_decay=1e-5 \
#        --grad_acc=1 \
#        --nworkers=12 \
#        --data_dir=$DATA_DIR \
#        --log_dir='/mnt/fsx1/bev_baseline/logs_nuscenes_bevseg' \
#        --ckpt_dir='/mnt/fsx1/bev_baseline/checkpoints/' \
#        --keep_latest=1 \
#        --ignore_load='' \
#        --load_step=False \
#        --load_optimizer=False \
#        --resolution_scale=2 \
#        --rand_flip=True \
#        --ncams=6 \
#        --nsweeps=3 \
#        --encoder_type='res101' \
#        --use_radar=False \
#        --use_lidar=False \
#        --do_metaradar=False \
#        --do_rgbcompress=True \
#        --device='cuda:0' \
#        --device_ids=[0,1,2,3] \
#        # --device='cuda:4' \
#        # --device_ids=[4,5,6,7] \

# rgb06: same but update the command, and decay 1e-5 since that's what i had for rad (i expect similar to rad05 though)
# rgb07: same
# rgb08: 4e-4 instead of 3e-4
# ok those didn't win
# rgb09: 5e-4
# rgb10: 3e-4, decay 1e-7
# rgb11: same but resize 0.9,1.1 instead of 0.8,1.2 < a bit worse!
# rgb12: 0.8,1.2 again; lr 5e-4, decay 1e-6, use_scheduler=True (yikes how long was that off?) < 47.55780943879203
# rgb13: same but 4e-4 < 47.40533491090929
# rgb14: connect weight decay to fetch_optimizer; 4e-4 repeat < 47.437589519538776
# rgb15: 3e-4 < 47.33251224323072
# ok
# higher lr is better here, so maybe i can stick with 5e-4.
# i think the next productive thing the gpus could do is: blast through the ablations
# let's go through in order.
# first is resolution!
# rgb18: resolution_scale=0.5 (also set to 416 instead of 400) < 36.57758048736164
# rgb16: resolution_scale=1 < 42.36708138655802
# (rgb15) < 47.33251224323072 
# rgb17: resolution_scale=3 < 49.27465376716988
# batch size next, restoring 400 and res_scale=2
# rgb19: grad_acc=1 
# rgb20: grad_acc=2
# rgb21: grad_acc=1, B4
# rgb22: grad_acc=1, B2
# rgb23: grad_acc=1, B1
# rgb24: grad_acc=1, B1, 50k
# rgb25: grad_acc=1, B1, 100k
# cams
# rgb26: grad_acc=5, B8, 25k, refcam=Front < 47.40575639409854
# rgb27: grad_acc=5, B8, 25k, refcam=rand, 5/6 ncams=5 < 46.40957627870228
# python train_nuscenes_bevseg.py \
#        --exp_name="rgb26" \
#        --max_iters=25000 \
#        --log_freq=1000 \
#        --shuffle=True \
#        --dset='trainval' \
#        --do_val=True \
#        --val_freq=100 \
#        --save_freq=1000 \
#        --batch_size=8 \
#        --grad_acc=5 \
#        --lr=5e-4 \
#        --use_scheduler=True \
#        --weight_decay=1e-6 \
#        --nworkers=12 \
#        --data_dir=$DATA_DIR \
#        --log_dir='/mnt/fsx1/bev_baseline/logs_nuscenes_bevseg' \
#        --ckpt_dir='/mnt/fsx1/bev_baseline/checkpoints/' \
#        --keep_latest=1 \
#        --ignore_load='' \
#        --load_step=False \
#        --load_optimizer=False \
#        --resolution_scale=2 \
#        --rand_flip=True \
#        --ncams=6 \
#        --nsweeps=3 \
#        --encoder_type='res101' \
#        --do_rgbcompress=True \
#        --device='cuda:0' \
#        --device_ids=[0,1,2,3] 

# python train_nuscenes_bevseg.py \
#        --exp_name="rgb04" \
#        --max_iters=40000 \
#        --log_freq=1000 \
#        --shuffle=True \
#        --dset='trainval' \
#        --do_val=True \
#        --val_freq=100 \
#        --save_freq=1000 \
#        --batch_size=40 \
#        --batch_parallel=8 \
#        --lr=3e-4 \
#        --weight_decay=1e-7 \
#        --grad_acc=1 \
#        --nworkers=12 \
#        --data_dir=$DATA_DIR \
#        --log_dir='/mnt/fsx1/bev_baseline/logs_nuscenes_bevseg' \
#        --ckpt_dir='/mnt/fsx1/bev_baseline/checkpoints/' \
#        --keep_latest=3 \
#        --ignore_load='' \
#        --init_dir='/mnt/fsx1/bev_baseline/checkpoints/40_3e-4_rgb04_03:48:13' \
#        --ignore_load='' \
#        --load_step=True \
#        --load_optimizer=False \
#        --resolution_scale=2 \
#        --rand_flip=True \
#        --ncams=6 \
#        --nsweeps=3 \
#        --encoder_type='res101' \
#        --use_radar=False \
#        --use_lidar=False \
#        --do_metaradar=False \
#        --do_rgbcompress=True \
#        --device='cuda:4' \
#        --device_ids=[4,5,6,7] 


# rad: aiming for 55.7

# rad00: use scheduler, go 25k, use_radar=True
# rad01: use scheduler, go 25k, use_radar=True, use_metaradar=True
# idea: maybe with rad i need to go higher, like to 40k
# rad02: do not use scheduler; go 40k, partly to help learn the distance required
# idea: maybe ncams=5 is necessary for radar, to force the model to use it alone sometimes
# rad03: ncams=5
# rad04: ncams=5; flip rad_occ_mem (bugfix)
# rad05: ncams=6
# rad06: ncams=6; 30k; use_scheduler=True
# rad07: X flip and Z flip
# rad08: X flip and Z flip; flip back in reverse order
# rad09: X flip and Z flip; flip back in reverse order; account for Y being gone
# rad10: 25k, ncams=5,
# rad11: 25k, ncams=5, aug 0.8,1.2
# rad12; init from 25k 40_3e-4_rad10_06:18:56, go 50k (because it was still improving, and also i widened those augs for rad11)
# rad13; init from 25k 40_3e-4_rad10_06:18:56, go 80k, weight decay 1e-6 instead of 1e-7
# idea:
# we are essentially dealing with a train/val gap here, and the rad model can get ~70 iou on train
# if you do another reasonable aug, you should add another percent
# e.g., slight translation augs in 3d
# > unfortunately, the label creation makes this difficult
# rad14: init from 50k 40_3e-4_rad12_05:42:42, go 100k, weight decay 1e-6, ncams=6 instead of 5
# rad15: same but go to 200k (tried to init from rad13 but did not)
# rad16: init from 97k 40_3e-4_rad14_05:00:47, and *disable filters*
# rad17: train from scratch 25k with scheduler, decay 1e-7 < unable to open shared memory object </torch_77553_1139403377> in read-write mode
# rad18: train from scratch 25k with scheduler, decay 1e-7
# 40_3e-4_rad16_03:12:44 at 122k is apparently 56.2; saved this for eval later
# rad19 similar to rad18 but train 50k
# rad20 similar but weight decay 1e-6
# python train_nuscenes_bevseg.py \
#        --exp_name="rad20" \
#        --max_iters=50000 \
#        --log_freq=1000 \
#        --shuffle=True \
#        --dset='trainval' \
#        --do_val=True \
#        --val_freq=100 \
#        --save_freq=1000 \
#        --batch_size=40 \
#        --batch_parallel=8 \
#        --lr=3e-4 \
#        --use_scheduler=True \
#        --weight_decay=1e-4 \
#        --grad_acc=1 \
#        --nworkers=12 \
#        --data_dir=$DATA_DIR \
#        --log_dir='/mnt/fsx1/bev_baseline/logs_nuscenes_bevseg' \
#        --ckpt_dir='/mnt/fsx1/bev_baseline/checkpoints/' \
#        --keep_latest=3 \
#        --resolution_scale=2 \
#        --rand_flip=True \
#        --ncams=6 \
#        --nsweeps=3 \
#        --encoder_type='res101' \
#        --use_radar=True \
#        --use_lidar=False \
#        --do_metaradar=True \
#        --do_rgbcompress=True \
#        --device='cuda:0' \
#        --device_ids=[0,1,2,3]

       # --init_dir='/mnt/fsx1/bev_baseline/checkpoints/40_3e-4_rad14_05:00:47' \
	   # --load_step=True \

# gosh, such a wide and expanding train/val gap. it should be so easy to pull down train while increasing val
# rad21 25k, weight_decay 1e-4, 24 workers instead of 12, since prev i had readtime < 
# rad22 25k, alt method of accumulating gradients; decay 1e-5 < killed accidentally
# killed accidentally
# --dset="mini" \

# use_metaradar was disabled accidentally (due to rename)
# turning it on now
# rad23: 25k, lr 3e-4
# rad24: 25k, lr 4e-4
# rad25: 25k, lr 5e-4
# rad26: 25k, lr 6e-4
# rad27: 25k, lr 5e-4 (repeat for std)
# ablations
# rad28: use_metaradar=False
# rad29: use_metaradar=true, filters=true
# rad30: use_metaradar=true, filters=false, nsweeps=1
# rad31: use_metaradar=true, filters=false, nsweeps=5
python train_nuscenes_bevseg.py \
       --exp_name="rad31" \
       --max_iters=25000 \
       --log_freq=1000 \
       --batch_size=8 \
       --grad_acc=5 \
       --lr=5e-4 \
       --use_scheduler=True \
       --weight_decay=1e-5 \
       --nworkers=24 \
       --data_dir=$DATA_DIR \
       --log_dir='/mnt/fsx1/bev_baseline/logs_nuscenes_bevseg' \
       --ckpt_dir='/mnt/fsx1/bev_baseline/checkpoints/' \
       --keep_latest=3 \
       --resolution_scale=2 \
       --ncams=6 \
       --nsweeps=5 \
       --encoder_type='res101' \
       --use_radar=True \
       --use_metaradar=True \
       --use_radar_filters=False \
       --device='cuda:4' \
       --device_ids=[4,5,6,7]


# lid00 lidar, 5e-4
# lid01 lidar, 4e-4
# lid02 lidar, 3e-4
# lid03 lidar, 6e-4
# lid04 lidar, 6e-4 using 0.9,1.1
# python train_nuscenes_bevseg.py \
#        --exp_name="lid04" \
#        --max_iters=25000 \
#        --log_freq=1000 \
#        --batch_size=8 \
#        --grad_acc=5 \
#        --lr=6e-4 \
#        --use_scheduler=True \
#        --weight_decay=1e-5 \
#        --nworkers=24 \
#        --data_dir=$DATA_DIR \
#        --log_dir='/mnt/fsx1/bev_baseline/logs_nuscenes_bevseg' \
#        --ckpt_dir='/mnt/fsx1/bev_baseline/checkpoints/' \
#        --keep_latest=3 \
#        --resolution_scale=2 \
#        --ncams=6 \
#        --nsweeps=3 \
#        --encoder_type='res101' \
#        --use_lidar=True \
#        --device='cuda:4' \
#        --device_ids=[4,5,6,7]
