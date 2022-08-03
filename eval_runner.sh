#!/bin/bash

DATA_DIR="/mnt/fsx/nuscenes"
# there should be ${DATA_DIR}/trainval/v1.0-trainval

# rgb00: default, on gpus 0-3
# rgb01: flip=True, on gpus 4-7

# python eval_nuscenes_bevseg.py \
#        --exp_name="rgb00" \
#        --data_dir=$DATA_DIR \
#        --log_dir='logs_eval_nuscenes_bevseg' \
#        --init_dir='checkpoints/40_3e-4_rgb00_00:52:38' \
       # --device='cuda:0' \
       # --device_ids=[0,1,2,3] \
#        # --device_ids=[0,1,2,3] \

# 20k: 0.4644063812303654
# 19k: 0.4686688397908696

# 40_3e-5_rgb02_00:51:48 23k: 39.0 ? what the hell
# --init_dir='checkpoints/40_3e-5_rgb02_00:51:48' \
   # --init_dir='checkpoints/40_3e-4_rgb01_01:13:43' \
    # oh oh
# it's because i updated the cropping
# 40_3e-5_rgb02_00:51:48 23k and feed new resize lim stuff: 
 # --init_dir='checkpoints/40_3e-4_rgb01_01:13:43' \
     # --dset='mini' \
     # after updating the settings slightly:
# rgb02: 44.7
# ok garbage. let's wait and see the result of the next nets.

# rgb04: 25k: 45.79609990531495
# rgb05: 25k: 47.1787797626123
# 8x5_3e-4_rgb06_19:35:12: 46.463983336823304
# 8x5_3e-4_rgb07_20:26:00: 46.316773315601644
# 8x5_4e-4_rgb08_20:27:12: 46.4973733482569
# 8x5_5e-4_rgb09_19:15:31: 46.236855964350525
# 8x5_3e-4_rgb10_06:24:04: 46.540322982356514
# 8x5_3e-4_rgb11_02:54:25: 46.388311827282486
# gosh maybe none of these used the scheduler
# ...rewrote the eval
# rgb11 re-eval: 46.40237103956216 < ok match
# 8x5_5e-4_rgb12_22:43:46: 47.55780943879203
# 8x5_4e-4_rgb13_22:45:52: 47.40533491090929
# alt eval, where we return the iou every time
# rgb13 re-eval: iou_ev 47.4, iou_ev2 46.6
# so the global iou is helping
# but let's check again sanja's repo
# yup. https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/tools.py#L269
# 'iou': total_intersect / total_union,
# so let's stick with first eval
# 8x5_4e-4s_rgb14_21:14:05: 47.437589519538776 
# 8x5_3e-4s_rgb15_21:20:26: 47.33251224323072
# now for ablations
# resolution
# 8x5_5e-4s_rgb16_04:10:41, scale 1: 42.36708138655802
# 8x5_5e-4s_rgb17_04:11:43, scale 3 (400): 
# 8x5_5e-4s_rgb18_19:23:55, scale 0.5, also set to 416 instead of 400: 36.57758048736164
# batchsize
# 8_5e-4s_rgb19_19:26:58: 44.480124184301005
# 8x2_5e-4s_rgb20_19:30:31: 46.371658090584
# 4_5e-4s_rgb21_04:03:00
# 2_5e-4s_rgb22_04:06:15
# 1_5e-4s_rgb23_04:06:46
# 1_5e-4s_rgb24_05:57:24
# 1_5e-4s_rgb25_05:58:15
# cameras
# 8x5_5e-4s_rgb26_17:52:44
# 8x5_5e-4s_rgb27_17:15:55
python eval_nuscenes_bevseg.py \
       --exp_name="rgb27" \
       --data_dir=$DATA_DIR \
       --log_dir='/mnt/fsx1/bev_baseline/logs_eval_nuscenes_bevseg' \
       --init_dir='/mnt/fsx1/bev_baseline/checkpoints/8x5_5e-4s_rgb27_17:15:55' \
       --resolution_scale=2 \
       --device='cuda:4' \
       --device_ids=[4,5,6,7]


# rad04 40k: 53.36171725828961
# rad09 30k: 53.189459455580256
# 40_3e-4_rad18_17:41:36 25k: 55.38595757475545
# 40_3e-4_rad16_03:12:44 at 122k: 55.13208535120829
# 40_3e-4_rad19_19:09:57 at 50k: 55.42852276637131
# 40_3e-4_rad20_19:13:17 at 50k: 55.55791382662683
# 40_3e-4_rad20_19:13:17 at 49k: 55.55368349387456
# note rgb05 had resize_lim 0.9-1.1; here it's 0.8-1.2
# 40_3e-4_rad21_21:16:12 25k: 55.00816918777085
# this was weight decay 1e-4
# but there was no effect on train compared to rad18, so i think this was just bad luck in the ckpt 
# 8x5_3e-4_rad22_04:04:53 22k (killed by accident, use_metaradar=False by accident): 53.986605660376966 < not bad for a no meta model
# 8x5_3e-4_rad23_04:04:53 25k: 55.14818582340928
# 8x5_4e-4_rad24_22:39:20 25k: 55.5924135087522
# ok cool. let's increase one more time and go
# --init_dir='/mnt/fsx1/bev_baseline/checkpoints/8x5_4e-4_rad24_22:39:20' \

# 8x5_5e-4_rad25_18:55:34: 55.79075749280007
# 8x5_6e-4_rad25_18:56:01: 55.620367287757944
# 8x5_5e-4_rad27_19:26:17: 55.71121026326078

# python eval_nuscenes_bevseg.py \
#        --exp_name="rad27" \
#        --data_dir=$DATA_DIR \
#        --log_dir='/mnt/fsx1/bev_baseline/logs_eval_nuscenes_bevseg' \
#        --init_dir='/mnt/fsx1/bev_baseline/checkpoints/8x5_5e-4_rad27_19:26:17' \
#        --use_radar=True \
#        --use_metaradar=True \
#        --device='cuda:4' \
#        --device_ids=[4,5,6,7] 
#        # --device='cuda:0' \
#        # --device_ids=[0,1,2,3] 
#        # --init_dir='/mnt/fsx1/bev_baseline/checkpoints/40_3e-4_rad18_17:41:36' \


# 8x5_5e-4_lid00_17:21:49: 63.89539955852593
# 8x5_4e-4_lid01_20:47:23: 63.95550985705564
# 8x5_3e-4_lid02_19:13:50: 63.468334914611006
# 8x5_6e-4_lid03_06:20:19: 64.17614962980704
# 8x5_6e-4_lid04_02:55:17: 63.87449500151892
# python eval_nuscenes_bevseg.py \
#        --exp_name="lid04" \
#        --data_dir=$DATA_DIR \
#        --log_dir='/mnt/fsx1/bev_baseline/logs_eval_nuscenes_bevseg' \
#        --init_dir='/mnt/fsx1/bev_baseline/checkpoints/8x5_6e-4_lid04_02:55:17' \
#        --use_lidar=True \
#        --device='cuda:0' \
#        --device_ids=[0,1,2,3] 
#        # --device='cuda:0' \
#        # --device_ids=[0,1,2,3] 
#        # --init_dir='/mnt/fsx1/bev_baseline/checkpoints/40_3e-4_rad18_17:41:36' \
