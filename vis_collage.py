import numpy as np
import os
import imageio
import cv2
import torch
import torch.nn.functional as F

vid = 14
folder_name = "/zfsauton2/home/zhaoyuaf/map3d/sample_vis_{0}".format(vid)
savefolder_name = "/zfsauton2/home/zhaoyuaf/map3d/sample_vis_{0}_collate".format(vid)
os.makedirs(savefolder_name, exist_ok=True)
T = 40
seg_h, seg_w = 600, 600
rgb_h, rgb_w = 187, 400
assert(seg_w * 2 == rgb_w * 3)

for t in range(T):
    print(t)
    seg_g = imageio.imread(os.path.join(folder_name, "seg_gt_%03d.png" % t))
    seg_g = cv2.resize(seg_g.astype(np.uint8), (seg_w, seg_h))
    seg_g = np.flip(seg_g, 0)

    seg_e = imageio.imread(os.path.join(folder_name, "seg_et_%03d.png" % t))
    seg_e = cv2.resize(seg_e.astype(np.uint8), (seg_w, seg_h))
    seg_e = np.flip(seg_e, 0)

    cam0 = imageio.imread(os.path.join(folder_name, "cam0_rgb_%03d.png" % t))
    cam0 = cv2.resize(cam0.astype(np.uint8), (rgb_w, rgb_h))

    cam1 = imageio.imread(os.path.join(folder_name, "cam1_rgb_%03d.png" % t))
    cam1 = cv2.resize(cam1.astype(np.uint8), (rgb_w, rgb_h))

    cam2 = imageio.imread(os.path.join(folder_name, "cam2_rgb_%03d.png" % t))
    cam2 = cv2.resize(cam2.astype(np.uint8), (rgb_w, rgb_h))

    cam3 = imageio.imread(os.path.join(folder_name, "cam3_rgb_%03d.png" % t))
    cam3 = cv2.resize(cam3.astype(np.uint8), (rgb_w, rgb_h))

    cam4 = imageio.imread(os.path.join(folder_name, "cam4_rgb_%03d.png" % t))
    cam4 = cv2.resize(cam4.astype(np.uint8), (rgb_w, rgb_h))

    cam5 = imageio.imread(os.path.join(folder_name, "cam5_rgb_%03d.png" % t))
    cam5 = cv2.resize(cam5.astype(np.uint8), (rgb_w, rgb_h))

    # collect into collate
    collate_t = np.zeros((seg_h + rgb_h*2, seg_w * 2, 3))
    collate_t[:seg_h, :seg_w] = seg_e
    collate_t[:seg_h, seg_w:] = seg_g
    collate_t[seg_h:seg_h+rgb_h, :rgb_w] = cam1
    collate_t[seg_h:seg_h+rgb_h, rgb_w:rgb_w*2] = cam0
    collate_t[seg_h:seg_h+rgb_h, rgb_w*2:] = cam2
    collate_t[seg_h+rgb_h:seg_h+rgb_h*2, :rgb_w] = cam5
    collate_t[seg_h+rgb_h:seg_h+rgb_h*2, rgb_w:rgb_w*2] = cam4
    collate_t[seg_h+rgb_h:seg_h+rgb_h*2, rgb_w*2:] = cam3

    collate_t_name = os.path.join(savefolder_name, "collate_%03d.png" % t)
    imageio.imsave(collate_t_name, collate_t.astype(np.uint8))
