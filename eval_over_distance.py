import os
import time
import argparse
import numpy as np
import saverloader
from fire import Fire
from nets.segnet import Segnet
import utils.misc
import utils.improc
import utils.vox
import random
import nuscenesdataset
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F

random.seed(125)
np.random.seed(125)

scene_centroid_x = 0.0
scene_centroid_y = 1.0
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 200, 8, 200

max_dist = 71
num_bins = 10
bin_size = max_dist/num_bins


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = utils.basic.reduce_masked_mean(loss, valid)
        return loss

def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = utils.basic.reduce_masked_mean(mse_loss, pos_mask*valid)
    neg_loss = utils.basic.reduce_masked_mean(mse_loss, neg_mask*valid)
    loss = (pos_loss + neg_loss)*0.5
    return loss

def balanced_ce_loss(out, target, valid):
    B, N, H, W = out.shape
    total_loss = torch.tensor(0.0).to(out.device)
    normalizer = 0
    NN = valid.shape[1]
    assert(NN==1)
    for n in range(N):
        out_ = out[:,n]
        tar_ = target[:,n]
        val_ = valid[:,0]

        pos = tar_.gt(0.99).float()
        neg = tar_.lt(0.95).float()
        label = pos*2.0 - 1.0
        a = -label * out_
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))
        if torch.sum(pos*val_) > 0:
            pos_loss = utils.basic.reduce_masked_mean(loss, pos*val_)
            neg_loss = utils.basic.reduce_masked_mean(loss, neg*val_)
            total_loss += (pos_loss + neg_loss)*0.5
            normalizer += 1
        else:
            total_loss += loss.mean()
            normalizer += 1
    return total_loss / normalizer
    
def balanced_occ_loss(pred, occ, free):
    pos = occ.clone()
    neg = free.clone()

    label = pos*2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

    mask_ = (pos+neg>0.0).float()

    pos_loss = utils.basic.reduce_masked_mean(loss, pos)
    neg_loss = utils.basic.reduce_masked_mean(loss, neg)

    balanced_loss = pos_loss + neg_loss

    return balanced_loss

def run_model(model, loss_fn, d, device='cuda:0', sw=None):
    metrics = {}
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    imgs, rots, trans, intrins, pts0, extra0, pts, extra, lrtlist_velo, vislist, tidlist, scorelist, seg_bev_g, valid_bev_g, center_bev_g, offset_bev_g, radar_data, egopose = d

    B0,T,S,C,H,W = imgs.shape
    assert(T==1)

    # eliminate the time dimension
    imgs = imgs[:,0]
    rots = rots[:,0]
    trans = trans[:,0]
    intrins = intrins[:,0]
    pts0 = pts0[:,0]
    extra0 = extra0[:,0]
    pts = pts[:,0]
    extra = extra[:,0]
    lrtlist_velo = lrtlist_velo[:,0]
    vislist = vislist[:,0]
    tidlist = tidlist[:,0]
    scorelist = scorelist[:,0]
    seg_bev_g = seg_bev_g[:,0]
    valid_bev_g = valid_bev_g[:,0]
    center_bev_g = center_bev_g[:,0]
    offset_bev_g = offset_bev_g[:,0]
    radar_data = radar_data[:,0]
    egopose = egopose[:,0]
    
    origin_T_velo0t = egopose.to(device) # B,T,4,4

    lrtlist_velo = lrtlist_velo.to(device)
    scorelist = scorelist.to(device)

    rgb_camXs = imgs.float().to(device)
    rgb_camXs = rgb_camXs - 0.5 # go to -0.5, 0.5

    seg_bev_g = seg_bev_g.to(device)
    valid_bev_g = valid_bev_g.to(device)
    center_bev_g = center_bev_g.to(device)
    offset_bev_g = offset_bev_g.to(device)

    xyz_velo0 = pts.to(device).permute(0, 2, 1)
    rad_data = radar_data.to(device).permute(0, 2, 1) # B, R, 19
    xyz_rad = rad_data[:,:,:3]
    meta_rad = rad_data[:,:,3:]

    B, S, C, H, W = rgb_camXs.shape
    B, V, D = xyz_velo0.shape

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    mag = torch.norm(xyz_velo0, dim=2)
    xyz_velo0 = xyz_velo0[:,mag[0]>1]
    xyz_velo0_bak = xyz_velo0.clone()

    intrins_ = __p(intrins)
    pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
    pix_T_cams = __u(pix_T_cams_)

    velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
    cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))
    
    cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
    camXs_T_cam0 = __u(utils.geom.safe_inverse(__p(cam0_T_camXs)))
    cam0_T_camXs_ = __p(cam0_T_camXs)
    camXs_T_cam0_ = __p(camXs_T_cam0)
    
    xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:,0], xyz_velo0)
    rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:,0], xyz_rad)

    lrtlist_cam0 = utils.geom.apply_4x4_to_lrtlist(cams_T_velo[:,0], lrtlist_velo)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)
    
    V = xyz_velo0.shape[1]

    occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X, assert_cube=False)
    rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
    metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, meta_rad, Z, Y, X, assert_cube=False)

    if not (model.module.use_radar or model.module.use_lidar):
        in_occ_mem0 = None
    elif model.module.use_lidar:
        assert(model.module.use_radar==False) # either lidar or radar, not both
        assert(model.module.use_metaradar==False) # either lidar or radar, not both
        in_occ_mem0 = occ_mem0
    elif model.module.use_radar and model.module.use_metaradar:
        in_occ_mem0 = metarad_occ_mem0
    elif model.module.use_radar:
        in_occ_mem0 = rad_occ_mem0
    elif model.module.use_metaradar:
        assert(False) # cannot use_metaradar without use_radar

    cam0_T_camXs = cam0_T_camXs

    lrtlist_cam0_g = lrtlist_cam0

    _, feat_bev_e, seg_bev_e, center_bev_e, offset_bev_e = model(
            rgb_camXs=rgb_camXs,
            pix_T_cams=pix_T_cams,
            cam0_T_camXs=cam0_T_camXs,
            vox_util=vox_util,
            rad_occ_mem0=in_occ_mem0)

    ce_loss = loss_fn(seg_bev_e, seg_bev_g, valid_bev_g)
    center_loss = balanced_mse_loss(center_bev_e, center_bev_g)
    offset_loss = torch.abs(offset_bev_e-offset_bev_g).sum(dim=1, keepdim=True)
    offset_loss = utils.basic.reduce_masked_mean(offset_loss, seg_bev_g*valid_bev_g)

    ce_factor = 1 / torch.exp(model.module.ce_weight)
    ce_loss = 10.0 * ce_loss * ce_factor
    ce_uncertainty_loss = 0.5 * model.module.ce_weight

    center_factor = 1 / (2*torch.exp(model.module.center_weight))
    center_loss = center_factor * center_loss
    center_uncertainty_loss = 0.5 * model.module.center_weight

    offset_factor = 1 / (2*torch.exp(model.module.offset_weight))
    offset_loss = offset_factor * offset_loss
    offset_uncertainty_loss = 0.5 * model.module.offset_weight

    total_loss += ce_loss
    total_loss += center_loss
    total_loss += offset_loss
    total_loss += ce_uncertainty_loss
    total_loss += center_uncertainty_loss
    total_loss += offset_uncertainty_loss

    seg_bev_e_round = torch.sigmoid(seg_bev_e).round()
    intersection = (seg_bev_e_round*seg_bev_g*valid_bev_g).sum()
    union = ((seg_bev_e_round+seg_bev_g)*valid_bev_g).clamp(0,1).sum()

    inter_map = seg_bev_e_round*seg_bev_g*valid_bev_g
    union_map = ((seg_bev_e_round+seg_bev_g)*valid_bev_g).clamp(0,1)


    coords_mem = utils.basic.gridcloud3d(B,Z,1,X) # B,-1,3
    coords_cam = vox_util.Mem2Ref(coords_mem, Z, Y, X) # B,-1,3
    dists_cam = torch.norm(coords_cam, dim=2) # B,-1
    dist_map = dists_cam.reshape(1, 1, Z, X)
    utils.basic.print_stats('dist_map', dist_map)
    
    # max_dist = torch.max(dist_map)

    inter_map_ = inter_map.reshape(-1)
    union_map_ = union_map.reshape(-1)
    dist_map_ = dist_map.reshape(-1)

    inters_ = torch.zeros(num_bins, dtype=torch.float32, device='cuda')
    unions_ = torch.zeros(num_bins, dtype=torch.float32, device='cuda')
    
    for bi in range(num_bins):
        bin_min = bin_size*bi
        bin_max = bin_size*(bi+1)
        print('considering %.1f to %.1f' % (bin_min, bin_max))
        inds = (dist_map_ > bin_min) & (dist_map_ < bin_max)
        inter_ = inter_map_[inds]
        union_ = union_map_[inds]
        inters_[bi] = inter_.sum()
        unions_[bi] = union_.sum()
        
    # utils.basic.print_('inters_', inters_)
    # utils.basic.print_('unions_', unions_)
    
    metrics['inters_'] = inters_.cpu().numpy()
    metrics['unions_'] = unions_.cpu().numpy()
    
    metrics['intersection'] = intersection.item()
    metrics['union'] = union.item()
    metrics['ce_loss'] = ce_loss.item()
    metrics['center_loss'] = center_loss.item()
    metrics['offset_loss'] = offset_loss.item()

    if sw is not None and sw.save_this:
        if model.module.use_radar or model.module.use_lidar:
            sw.summ_occ('0_inputs/rad_occ_mem0', rad_occ_mem0)
        sw.summ_occ('0_inputs/occ_mem0', occ_mem0)
        sw.summ_rgb('0_inputs/rgb_camXs', torch.cat(rgb_camXs[0:1].unbind(1), dim=-1))

        sw.summ_oned('2_outputs/seg_bev_g', seg_bev_g * (0.5+valid_bev_g*0.5), norm=False)
        sw.summ_oned('2_outputs/valid_bev_g', valid_bev_g, norm=False)
        sw.summ_oned('2_outputs/seg_bev_e', torch.sigmoid(seg_bev_e).round(), norm=False, frame_id=iou.item())
        sw.summ_oned('2_outputs/seg_bev_e_soft', torch.sigmoid(seg_bev_e), norm=False)

        sw.summ_oned('2_outputs/center_bev_g', center_bev_g, norm=False)
        sw.summ_oned('2_outputs/center_bev_e', center_bev_e, norm=False)

        sw.summ_flow('2_outputs/offset_bev_e', offset_bev_e, clip=10)
        sw.summ_flow('2_outputs/offset_bev_g', offset_bev_g, clip=10)

    return total_loss, metrics
    
def main(
        exp_name='eval',
        # val/test
        log_freq=100,
        shuffle=False,
        dset='trainval', # we will just use val
        batch_size=8,
        nworkers=12,
        # data/log/load directories
        data_dir='../nuscenes/',
        log_dir='logs_eval_nuscenes_bevseg',
        init_dir='checkpoints/rgb_model',
        ignore_load=None,
        # data
        res_scale=2,
        ncams=6,
        nsweeps=3,
        # model
        encoder_type='res101',
        use_radar=False,
        use_radar_filters=False,
        use_lidar=False,
        use_metaradar=False,
        do_rgbcompress=True,
        # cuda
        device_ids=[4,5,6,7],
):
    B = batch_size
    assert(B % len(device_ids) == 0) # batch size must be divisible by number of gpus

    device = 'cuda:%d' % device_ids[0]
    
    ## autogen a name
    model_name = "%s" % init_dir.split('/')[-1]
    model_name += "_%d" % B
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    # set up logging
    writer_ev = SummaryWriter(os.path.join(log_dir, model_name + '/ev'), max_queue=10, flush_secs=60)

    # set up dataloader
    final_dim = (int(224 * res_scale), int(400 * res_scale))
    print('resolution:', final_dim)
    
    data_aug_conf = {
        'final_dim': final_dim,
        'H': 900, 'W': 1600,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'ncams': ncams,
    }
    _, val_dataloader = nuscenesdataset.compile_data(
        dset,
        data_dir,
        data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py,
        bounds=bounds,
        res_3d=(Z,Y,X),
        bsz=B,
        nworkers=nworkers,
        shuffle=shuffle,
        use_radar_filters=use_radar_filters,
        seqlen=1, # we do not load a temporal sequence here, but that can work with this dataloader
        nsweeps=nsweeps,
        do_shuffle_cams=False,
        get_tids=True,
    )
    val_iterloader = iter(val_dataloader)

    max_iters = len(val_dataloader) # determine iters by length of dataset

    # set up model & seg loss
    seg_loss_fn = SimpleLoss(2.13).to(device)
    model = Segnet(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, encoder_type=encoder_type)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    parameters = list(model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params', total_params)

    # load checkpoint
    _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
    global_step = 0
    requires_grad(parameters, False)
    model.eval()

    # logging pools. pool size should be larger than max_iters
    n_pool = 10000
    loss_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    time_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    ce_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    center_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    offset_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    iou_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    itime_pool_ev = utils.misc.SimplePool(n_pool, version='np')
    assert(n_pool > max_iters)

    inters_over_distance = np.zeros(num_bins)
    unions_over_distance = np.zeros(num_bins)
    
    intersection = 0
    union = 0
    while global_step < max_iters:
        global_step += 1

        iter_start_time = time.time()
        read_start_time = time.time()

        sw_ev = utils.improc.Summ_writer(
            writer=writer_ev,
            global_step=global_step,
            log_freq=log_freq,
            fps=2,
            scalar_freq=int(log_freq/2),
            just_gif=True)
        sw_ev.save_this = False
        
        try:
            sample = next(val_iterloader)
        except StopIteration:
            break

        read_time = time.time()-read_start_time
            
        with torch.no_grad():
            total_loss, metrics = run_model(model, seg_loss_fn, sample, device, sw_ev)

        intersection += metrics['intersection']
        union += metrics['union']

        inters_over_distance += metrics['inters_']
        unions_over_distance += metrics['unions_']

        sw_ev.summ_scalar('pooled/iou_ev', intersection/union)
        
        loss_pool_ev.update([total_loss.item()])
        sw_ev.summ_scalar('pooled/total_loss', loss_pool_ev.mean())
        sw_ev.summ_scalar('stats/total_loss', total_loss.item())

        ce_pool_ev.update([metrics['ce_loss']])
        sw_ev.summ_scalar('pooled/ce_loss', ce_pool_ev.mean())
        sw_ev.summ_scalar('stats/ce_loss', metrics['ce_loss'])
        
        center_pool_ev.update([metrics['center_loss']])
        sw_ev.summ_scalar('pooled/center_loss', center_pool_ev.mean())
        sw_ev.summ_scalar('stats/center_loss', metrics['center_loss'])

        offset_pool_ev.update([metrics['offset_loss']])
        sw_ev.summ_scalar('pooled/offset_loss', offset_pool_ev.mean())
        sw_ev.summ_scalar('stats/offset_loss', metrics['offset_loss'])

        iter_time = time.time()-iter_start_time
        
        time_pool_ev.update([iter_time])
        sw_ev.summ_scalar('pooled/time_per_batch', time_pool_ev.mean())
        sw_ev.summ_scalar('pooled/time_per_el', time_pool_ev.mean()/float(B))

        print('%s; step %06d/%d; rtime %.2f; itime %.2f (%.2f ms); loss %.5f; iou_ev %.1f' % (
            model_name, global_step, max_iters, read_time, iter_time, 1000*time_pool_ev.mean(),
            total_loss.item(), 100*intersection/union))

        ious_over_distance = inters_over_distance / np.maximum(unions_over_distance, 1.0)
        print('ious_over_distance', 100*ious_over_distance)
        
    print('final %s mean iou' % dset, 100*intersection/union)

    # ious_over_distance = []
    # for i,u in zip(inters_over_distance, unions_over_distance):

    
    writer_ev.close()
            

if __name__ == '__main__':
    Fire(main)

