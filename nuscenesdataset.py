"""
Code adapted from https://github.com/nv-tlabs/lift-splat-shoot
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

# from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx
import torchvision
from functools import reduce
# from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.data_classes import PointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
from time import time

from utils.basic import print_, print_stats
import utils.py
import utils.geom
import itertools
import matplotlib.pyplot as plt

discard_invisible = False

def convert_egopose_to_matrix_numpy(egopose):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(egopose['rotation']).rotation_matrix
    translation = np.array(egopose['translation'])
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix

class LidarPointCloud(PointCloud):
    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 5

    @classmethod
    def from_file(cls, file_name: str) -> 'LidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """
        
        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)
        
        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :cls.nbr_dims()]
        return cls(points.T)

def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt, ring_index) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    # points = np.zeros((5, 0))
    points = np.zeros((6, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # print('time_lag', time_lag)
        # print('new_points', new_points.shape)
        
        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points

def get_radar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt, ring_index) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    # import ipdb; ipdb.set_trace()
    
    # points = np.zeros((5, 0))
    points = np.zeros((19, 0)) # 18 plus one for time

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['RADAR_FRONT']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),inverse=True)

    # RadarPointCloud.disable_filters()
    RadarPointCloud.default_filters()

    # Aggregate current and previous sweeps.
    # from all radars 
    radar_chan_list = ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
    for radar_name in radar_chan_list:
        sample_data_token = sample_rec['data'][radar_name]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = RadarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                                Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            times = time_lag * np.ones((1, current_pc.nbr_points()))

            new_points = np.concatenate((current_pc.points, times), 0)
            points = np.concatenate((points, new_points), 1)

            # print('time_lag', time_lag)
            # print('new_points', new_points.shape)

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points




def ego_to_cam(points, rot, trans, intrins=None):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    if intrins is not None:
        points = intrins.matmul(points)
        points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran

class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))

denormalize_img_torch = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
))


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))
totorch_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
))
normalize_img_torch = torchvision.transforms.Compose((
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_val_info(model, valloader, loss_fn, device, use_tqdm=False, max_iters=None, use_lidar=False):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')

    t0 = time()
    
    loader = tqdm(valloader) if use_tqdm else valloader

    if max_iters is not None:
        counter = 0
    with torch.no_grad():
        for batch in loader:

            if max_iters is not None:
                counter += 1
                if counter > max_iters:
                    break

            if use_lidar:
                allimgs, rots, trans, intrins, post_rots, post_trans, pts, binimgs = batch
            else:
                allimgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
                
            preds = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device))
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds[:,0:1], binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds[:,0:1], binimgs)
            total_intersect += intersect
            total_union += union
    t1 = time()
    print('eval took %.2f seconds' % (t1-t0))

    model.train()

    if max_iters is not None:
        normalizer = counter
    else:
        normalizer = len(valloader.dataset)
        
    return {
        'total_loss': total_loss / normalizer,
        'iou': total_intersect / total_union,
    }


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

def add_ego2(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+1, W/2.],
        [4.084/2.+1, W/2.],
        [4.084/2.+1, -W/2.],
        [-4.084/2.+1, -W/2.],
    ])
    pts = (pts - bx) / dx
    # pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)

def fetch_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
        50.0, poly_names, line_names)

    return poly_names, line_names, lmap

def fetch_nusc_map2(rec, nusc_maps, nusc, scene2map, car_from_current):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    global_from_car = transform_matrix(egopose['translation'],
                                       Quaternion(egopose['rotation']), inverse=False)

    trans_matrix = reduce(np.dot, [global_from_car, car_from_current])

    rot = np.arctan2(trans_matrix[1,0], trans_matrix[0,0])
    center = np.array([trans_matrix[0,3], trans_matrix[1,3], np.cos(rot), np.sin(rot)])

    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    poly_names = ['drivable_area', 'road_segment', 'lane']
    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
        50.0, poly_names, line_names)

    return poly_names, line_names, lmap

def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf, nsweeps=1, include_extra=False, drop_rings=False, seqlen=1, cam_id=1, include_rgbs=True, get_tids=False, temporal_aug=False):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.nsweeps = nsweeps
        self.drop_rings = drop_rings
        self.include_extra = include_extra
        self.include_rgbs = include_rgbs

        self.seqlen = seqlen
        self.cam_id = cam_id
        
        self.scenes = self.get_scenes()
        # print('applying hack to use just first scene')
        # self.scenes = self.scenes[0:1]
        self.ixes = self.prepro()
        if temporal_aug:
            self.indices = self.get_indices_tempaug()
        else:
            self.indices = self.get_indices()

        self.get_tids = get_tids

        # print('ixes', self.ixes.shape)
        print('indices', self.indices.shape)

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        # self.fix_nuscenes_formatting()

        print(self)

    # def fix_nuscenes_formatting(self):
    #     """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
    #     stored in the nuScenes object.
    #     """
    #     # check if default file paths work
    #     rec = self.ixes[0]
    #     sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
    #     imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

    #     def find_name(f):
    #         d, fi = os.path.split(f)
    #         d, di = os.path.split(d)
    #         d, d0 = os.path.split(d)
    #         d, d1 = os.path.split(d)
    #         d, d2 = os.path.split(d)
    #         return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

    #     # adjust the image paths if needed
    #     if not os.path.isfile(imgname):
    #         print('adjusting nuscenes file paths')
    #         fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
    #         fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
    #         info = {}
    #         for f in fs:
    #             di, fi, fname = find_name(f)
    #             info[f'samples/{di}/{fi}'] = fname
    #         fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
    #         for f in fs:
    #             di, fi, fname = find_name(f)
    #             info[f'sweeps/{di}/{fi}'] = fname
    #         for rec in self.nusc.sample_data:
    #             if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
    #                 rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.seqlen):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                index += 38
                indices.append(current_indices)

        return np.asarray(indices)

    def get_indices_tempaug(self):
        indices = []
        t_patterns = None
        if self.seqlen == 1:
            return self.get_indices()
        elif self.seqlen == 2:
            # seq options: (t, t+1), (t, t+2)
            t_patterns = [[0,1], [0,2]]
        elif self.seqlen == 3:
            # seq options: (t, t+1, t+2), (t, t+1, t+3), (t, t+2, t+3)
            t_patterns = [[0,1,2], [0,1,3], [0,2,3]]
        elif self.seqlen == 5:
            t_patterns = [
                    [0,1,2,3,4], # normal
                    [0,1,2,3,5], [0,1,2,4,5], [0,1,3,4,5], [0,2,3,4,5], # 1 skip
                    # [1,0,2,3,4], [0,2,1,3,4], [0,1,3,2,4], [0,1,2,4,3], # 1 reverse
                    ]
        else:
            raise NotImplementedError("timestep not implemented")

        for index in range(len(self.ixes)):
            for t_pattern in t_patterns:
                is_valid_data = True
                previous_rec = None
                current_indices = []
                for t in t_pattern:
                    index_t = index + t
                    # going over the dataset size limit
                    if index_t >= len(self.ixes):
                        is_valid_data = False
                        break
                    rec = self.ixes[index_t]
                    # check if scene is the same
                    if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                        is_valid_data = False
                        break
                    
                    current_indices.append(index_t)
                    previous_rec = rec

                if is_valid_data:
                    indices.append(current_indices)
                    # indices.append(list(reversed(current_indices)))
                    # indices += list(itertools.permutations(current_indices))

        return np.asarray(indices)
    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if 'resize_lim' in self.data_aug_conf and self.data_aug_conf['resize_lim'] is not None:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            else:
                resize = self.data_aug_conf['resize_scale']
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims

            # crop_h = 46
            # crop_w = int(max(0, (resized_width - final_width) / 2))

            # crop_h = self.data_aug_conf['top_crop']
            # crop_h = int(max((newH - fH) / 4, (newH - fH) / 2)) # tilt to positive
            # crop_h = int(max(0, (newH - fH) / 2))
            # crop_w = int(max(0, (newW - fW) / 2))
            
            # crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            # crop_w = int(np.random.uniform(0, max(0, newW - fW)))

            # crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            
            crop_h = int(np.random.uniform(0, max(0, newH - fH)))
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            # print('crop', crop)
            
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            
            # crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            # crop_w = int(max(0, newW - fW) / 2)

            # # crop_h = self.data_aug_conf['top_crop']
            # crop_h = int(max(0, (newH - fH) / 2))
            # crop_w = int(max(0, (newW - fW) / 2))

            # crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_h = int(max(0, newH - fH) / 2)
            crop_w = int(max(0, newW - fW) / 2)
            

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams, include_rgbs=True):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            if include_rgbs:
                imgname = os.path.join(self.nusc.dataroot, samp['filename'])
                img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])



            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()

            # print('resize', resize)
            # intrin = utils.py.scale_intrinsics(intrin, resize, resize)
            intrin = utils.geom.scale_intrinsics(intrin.unsqueeze(0), resize, resize).squeeze(0)
            # print('crop', crop)
            # crop = (0,0,352,128)

            fx, fy, x0, y0 = utils.geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = utils.geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)
            

            if include_rgbs:
                img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                         resize=resize,
                                                         resize_dims=resize_dims,
                                                         crop=crop,
                                                         flip=flip,
                                                         rotate=rotate,
                                                         )
            
                # for convenience, make augmentation matrices 3x3
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2
                imgs.append(totorch_img(img))
            else:
                newW, newH = resize_dims
                imgs.append(torch.zeros((3, newH, newW), dtype=torch.float32))

            # imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            # post_rots.append(post_rot)
            # post_trans.append(post_tran)
        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins))
        # return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
        #         torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))


    def get_lidar_data(self, rec, nsweeps, include_extra=False):
        pts = get_lidar_data(self.nusc, rec,
                             nsweeps=nsweeps, min_distance=2.2)
        if include_extra:
            return torch.Tensor(pts)
        else:
            return torch.Tensor(pts)[:3]  # x,y,z

    def get_radar_data(self, rec, nsweeps, include_extra=False):
        pts = get_radar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)
        # if include_extra:
        #     return torch.Tensor(pts)
        # else:
        #     return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
                                # self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for ii, tok in enumerate(rec['anns']):
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            if discard_invisible and int(inst['visibility_token']) == 1:
                # filter invisible vehicles
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], ii+1.0)
        # utils.py.print_stats('img', img)

        return torch.Tensor(img).unsqueeze(0), torch.Tensor(convert_egopose_to_matrix_numpy(egopose))

    def get_seg_bev(self, lrtlist_cam, vislist):
        B, N, D = lrtlist_cam.shape
        assert(B==1)
        seg = np.zeros((self.Z, self.X))
        val = np.ones((self.Z, self.X))

        corners_cam = utils.geom.get_xyzlist_from_lrtlist(lrtlist_cam) # B, N, 8, 3
        y_cam = corners_cam[:,:,:,1] # y part; B, N, 8
        corners_mem = self.vox_util.Ref2Mem(corners_cam.reshape(B, N*8, 3), self.Z, self.Y, self.X).reshape(B, N, 8, 3)

        # take the xz part
        corners_mem = torch.stack([corners_mem[:,:,:,0], corners_mem[:,:,:,2]], dim=3) # B, N, 8, 2
        # corners_mem = corners_mem[:,:,:4] # take the bottom four
        
        for n in range(N):
            _, inds = torch.topk(y_cam[0,n], 4, largest=False)
            pts = corners_mem[0,n,inds].numpy().astype(np.int32) # 4, 2

            # if this messes in some later conditions,
            # the solution is to draw all combos
            pts = np.stack([pts[0],pts[1],pts[3],pts[2]])
            
            # pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(seg, [pts], n+1.0)
            
            if vislist[n]==0:
                # draw a black rectangle if it's invisible
                cv2.fillPoly(val, [pts], 0.0)

        return torch.Tensor(seg).unsqueeze(0), torch.Tensor(val).unsqueeze(0) # 1, Z, X

    def get_center_and_offset_bev(self, lrtlist_cam, seg_bev):
        B, N, D = lrtlist_cam.shape
        assert(B==1)

        lrtlist_mem = self.vox_util.apply_mem_T_ref_to_lrtlist(
            lrtlist_cam.clone(), self.Z, self.Y, self.X)
        clist_cam = utils.geom.get_clist_from_lrtlist(lrtlist_cam)
        lenlist, rtlist = utils.geom.split_lrtlist(lrtlist_cam) # B,N,3
        rlist_, tlist_ = utils.geom.split_rt(rtlist.reshape(B*N, 4, 4))
        # print_('rlist_', rlist_)
        # print_('tlist_', tlist_)

        x_vec = torch.zeros((B*N, 3), dtype=torch.float32, device=rlist_.device)
        x_vec[:, 0] = 1 # 0,0,1 
        # x_vec[:, 2] = 1 # 0,0,1 
        # print_('x_vec', x_vec)
        # import ipdb; ipdb.set_trace()
        x_rot = torch.matmul(rlist_, x_vec.unsqueeze(2)).squeeze(2)
        # print_('x_rot', x_rot)
        # rylist = torch.atan2(x_rot[:,0],x_rot[:,2]).reshape(B, N)
        # print_('rylist', rylist)

        # rylist = utils.geom.rotm2eul(rlist_)[1]
        # import ipdb; ipdb.set_trace()
        rylist = torch.atan2(x_rot[:, 0], x_rot[:, 2]).reshape(N)
        rylist = utils.geom.wrap2pi(rylist + np.pi/2.0)
        
        if False:
            # these are B x N x 3
            # print_('lenlist', lenlist)
            sizelist = (torch.max(lenlist, dim=2)[0]).clamp(min=2)
            # print_stats('sizelist', sizelist)
            sizelist = sizelist.clamp(min=4)
            radius = sizelist/4.0
        else:
            radius = 3
        center, offset = self.vox_util.xyz2circles(clist_cam, radius, self.Z, self.Y, self.X, soft=True, already_mem=False, also_offset=True)
        # B,N,1/3,Z,Y,X
        # print_stats('offset', offset)

        # zgrid, xgrid = torch.meshgrid(torch.arange(self.Z, dtype=torch.float), torch.arange(self.X, dtype=torch.float))
        # offset = torch.zeros((1, 2, self.Z, self.X), dtype=torch.float32)
        # masklist = 
        masklist = torch.zeros((1, N, 1, self.Z, 1, self.X), dtype=torch.float32)
        # sizelist = torch.zeros((1, N, 3, self.Z, 1, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            # masklist[0,n,0,:,0] = utils.improc.dilate2d(inst).squeeze()
            masklist[0,n,0,:,0] = (inst.squeeze() > 0.01).float()

        size_bev = torch.zeros((1, 3, self.Z, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            inst = inst.reshape(self.Z, self.X) > 0.01
            # print('size_bev[0:,inst]',
            # print('inst', inst.shape)
            # print('lenlist[0,n]', lenlist[0,n].shape)
            size_bev[0,:,inst] = lenlist[0,n].unsqueeze(1)

        ry_bev = torch.zeros((1, 1, self.Z, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            inst = inst.reshape(self.Z, self.X) > 0.01
            ry_bev[0,:,inst] = rylist[n]
            
        ycoord_bev = torch.zeros((1, 1, self.Z, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            inst = inst.reshape(self.Z, self.X) > 0.01
            ycoord_bev[0,:,inst] = tlist_[n,1] # y part
            
        # # center is B,N,Z,Y,X
        # # offset is B,N,3,Z,Y,X
        # masklist = self.vox_util.assemble_padded_obj_masklist(lrtlist_cam, torch.ones_like(lrtlist_cam[:,:,0]), self.Z, self.Y, self.X, additive_coeff=0.5)
        # # masklist is B,N,1,Z,Y,X
        # print_stats('masklist', masklist)

        offset = offset * masklist
        # print_stats('masked offset', offset)

        offset = torch.sum(offset, dim=1) # B,3,Z,Y,X
        # print_stats('sumN offset', offset)
        
        offset = torch.stack([offset[:,0], offset[:,2]], dim=1) # B,2,Z,Y,X, containing the xz offset
        # print_stats('xz offset', offset)
        min_offset = torch.min(offset, dim=3)[0] # B,2,Z,X
        max_offset = torch.max(offset, dim=3)[0] # B,2,Z,X
        offset = min_offset + max_offset
        # print_stats('minmax offset', offset)
        
        center = torch.max(center, dim=1, keepdim=True)[0] # B,1,Z,Y,X
        # offset = offset * center.unsqueeze(2) # B,1,
        center = torch.max(center, dim=3)[0] # max along Y; 1,Z,X
        return center.squeeze(0), offset.squeeze(0), size_bev.squeeze(0), ry_bev.squeeze(0), ycoord_bev.squeeze(0) # 1,Z,X; 2,Z,X; 3,Z,X; 1,Z,X

    def get_lrtlist(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
                                # self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        lrtlist = []
        boxlist = []
        vislist = []
        tidlist = []
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            # if discard_invisible and int(inst['visibility_token']) == 1:
            if int(inst['visibility_token']) == 1:
                vislist.append(torch.tensor(0.0)) # invisible
            else:
                vislist.append(torch.tensor(1.0)) # visible
                # # print('dropping invisible vehicle')
                # # filter invisible vehicles
                # continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            tidlist.append(inst['instance_token'])

            # print('rotation', inst['rotation'])
            r = box.rotation_matrix
            t = box.center
            l = box.wlh
            l = np.stack([l[1],l[0],l[2]])
            lrt = utils.py.merge_lrt(l, utils.py.merge_rt(r,t))
            lrt = torch.Tensor(lrt)
            lrtlist.append(lrt)
            ry, _, _ = Quaternion(inst['rotation']).yaw_pitch_roll
            # print('rx, ry, rz', rx, ry, rz)
            rs = np.stack([ry*0, ry, ry*0])
            box_ = torch.from_numpy(np.stack([t,l,rs])).reshape(9)
            # print('box_', box_)
            boxlist.append(box_)
        if len(lrtlist):
            lrtlist = torch.stack(lrtlist, dim=0)
            boxlist = torch.stack(boxlist, dim=0)
            vislist = torch.stack(vislist, dim=0)
            # tidlist = torch.stack(tidlist, dim=0)
        else:
            lrtlist = torch.zeros((0, 19))
            boxlist = torch.zeros((0, 9))
            vislist = torch.zeros((0))
            # tidlist = torch.zeros((0))
            tidlist = []

        return lrtlist, boxlist, vislist, tidlist

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.indices)
        # return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 0.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        
        XMIN, XMAX = -50, 50
        ZMIN, ZMAX = -50, 50
        # XMIN, XMAX = -25, 25
        # ZMIN, ZMAX = -25, 25
        YMIN, YMAX = -5, 5
        bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
        Z, Y, X = 200, 1, 200
        # Z, Y, X = 400, 1, 400
        # Z, Y, X = 128, 1, 128
        # Z, Y, X = 200*2, 1, 200*2
        self.vox_util = utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=scene_centroid,
            bounds=bounds,
            assert_cube=False)
        self.Z, self.Y, self.X = Z, Y, X

    def get_single_item(self, index, cams, cam_id=None):
        # print('index %d; cam_id' % index, cam_id)
        rec = self.ixes[index]
        
        # imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams, include_rgbs=self.include_rgbs)
        imgs, rots, trans, intrins = self.get_image_data(rec, cams, include_rgbs=self.include_rgbs)
        lidar_data = self.get_lidar_data(rec, nsweeps=self.nsweeps, include_extra=self.include_extra)
        binimg, egopose = self.get_binimg(rec)
        # print('egopose', egopose)

        if cam_id is None:
            if self.is_train:
                # randomly sample the ref cam
                cam_id = np.random.randint(1, len(cams))
            else:
                cam_id = self.cam_id
            
        # print('imgs', imgs.shape)
        # print('rots', rots.shape)

        imgs_ = imgs.clone()
        rots_ = rots.clone()
        trans_ = trans.clone()
        intrins_ = intrins.clone()

        # move the target cam_id to the zeroth slot
        imgs[0] = imgs_[cam_id].clone()
        imgs[cam_id] = imgs_[0].clone()

        rots[0] = rots_[cam_id].clone()
        rots[cam_id] = rots_[0].clone()

        trans[0] = trans_[cam_id].clone()
        trans[cam_id] = trans_[0].clone()

        intrins[0] = intrins_[cam_id].clone()
        intrins[cam_id] = intrins_[0].clone()

            
            
            
        # else:
        #     cam_id = self.cam_id
            
        radar_data = self.get_radar_data(rec, nsweeps=self.nsweeps)
        # print('radar_data', radar_data.shape)

        # print('lidar_data', lidar_data.shape)
        if self.include_extra:
            lidar_extra = lidar_data[3:]
            lidar_data = lidar_data[:3]
        else:
            lidar_extra = lidar_data[:2]*0
        # print('lidar_data', lidar_data.shape)

        lrtlist_, boxlist_, vislist_, tidlist_ = self.get_lrtlist(rec)
        # print('lrtlist', lrtlist.shape)
        N_ = lrtlist_.shape[0]
        # lrtlist_ is N, 19

        # import ipdb; ipdb.set_trace()

        if N_ > 0:
            
            velo_T_cam = utils.geom.merge_rt(rots, trans)
            cam_T_velo = utils.geom.safe_inverse(velo_T_cam)
            # print('lrtlist_', lrtlist_.shape)
            # lrtlist_cam = utils.geom.apply_4x4_to_lrt(cam_T_velo[cam_id:cam_id+1].repeat(N_, 1, 1), lrtlist_).unsqueeze(0)

            # note we index 0:1, since we already put refcam into zeroth position
            lrtlist_cam = utils.geom.apply_4x4_to_lrt(cam_T_velo[0:1].repeat(N_, 1, 1), lrtlist_).unsqueeze(0)
            # print('lrtlist_cam', lrtlist_cam.shape)

            seg_bev, valid_bev = self.get_seg_bev(lrtlist_cam, vislist_)
            # print('seg_bev', seg_bev.shape)
            
            center_bev, offset_bev, size_bev, ry_bev, ycoord_bev = self.get_center_and_offset_bev(lrtlist_cam, seg_bev)
            # print('center_bev', center_bev.shape)
            # print('size_bev', size_bev.shape)

            # # offset is 1,N,2,Z,X
            # offset = offset * seg_bev.reshape(1, 1, 1, self.Z, self.X)
            # offset_bev = torch.max(offset[0], dim=0)[0] # 2,Z,X
            
            
            # seg_mem = self.vox_util.assemble_padded_obj_masklist(
            #     lrtlist_cam, torch.ones_like(lrtlist_cam[:,:,0]), self.Z, self.Y, self.X, additive_coeff=0.25)
            # # print('seg_mem', seg_mem.shape)
            # seg_mem = torch.max(seg_mem, dim=1)[0]
            # # print('seg_mem', seg_mem.shape)
            # seg_bev = torch.max(seg_mem, dim=3)[0]
            # # print('seg_bev', seg_bev.shape)
            # seg_bev = seg_bev[0]
        else:
            seg_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            valid_bev = torch.ones((1, self.Z, self.X), dtype=torch.float32)
            center_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            offset_bev = torch.zeros((2, self.Z, self.X), dtype=torch.float32)
            size_bev = torch.zeros((3, self.Z, self.X), dtype=torch.float32)
            ry_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            ycoord_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)

        N = 150 # i've seen 103 before
        lrtlist = torch.zeros((N, 19), dtype=torch.float32)
        vislist = torch.zeros((N), dtype=torch.float32)
        scorelist = torch.zeros((N), dtype=torch.float32)
        lrtlist[:N_] = lrtlist_
        vislist[:N_] = vislist_
        scorelist[:N_] = 1

        # print('lrtlist', lrtlist.shape)
        # print('scorelist', scorelist.shape)
        # print('vislist', vislist.shape)

        
        # lidar is like 3, V, where V~=26k 

        # print('lidar_data', lidar_data.shape)

        # pad to some OK size;
        # it seems like ~26k is normal

        times = lidar_extra[2] # V
        inds = times==times[0]
        lidar0_data = lidar_data[:,inds]
        lidar0_extra = lidar_extra[:,inds]

        lidar0_data = np.transpose(lidar0_data)
        lidar0_extra = np.transpose(lidar0_extra)
        lidar_data = np.transpose(lidar_data)
        lidar_extra = np.transpose(lidar_extra)
        V = 30000*self.nsweeps
        if lidar_data.shape[0] > V:
            # # print('lidar_data.shape[0]', lidar_data.shape[0])
            # np.random.shuffle(lidar_data)

            assert(False) # i expect this to never happen
            lidar0_data = lidar0_data[:V//self.nsweeps]
            lidar0_extra = lidar0_extra[:V//self.nsweeps]
            lidar_data = lidar_data[:V]
            lidar_extra = lidar_extra[:V]
        elif lidar_data.shape[0] < V:
            lidar0_data = np.pad(lidar0_data,[(0,V//self.nsweeps-lidar0_data.shape[0]),(0,0)],mode='constant')
            lidar0_extra = np.pad(lidar0_extra,[(0,V//self.nsweeps-lidar0_extra.shape[0]),(0,0)],mode='constant')
            lidar_data = np.pad(lidar_data,[(0,V-lidar_data.shape[0]),(0,0)],mode='constant')
            lidar_extra = np.pad(lidar_extra,[(0,V-lidar_extra.shape[0]),(0,0)],mode='constant',constant_values=-1)
        lidar0_data = np.transpose(lidar0_data)
        lidar0_extra = np.transpose(lidar0_extra)
        lidar_data = np.transpose(lidar_data)
        lidar_extra = np.transpose(lidar_extra)

        if self.drop_rings:
            ring0 = lidar0_extra[1] # V
            # zero out a random number of rings
            num_rings = np.random.randint(2, 16)
            ring_inds = np.random.choice(32, size=num_rings, replace=False)
            # print('zeroing out:', ring_inds)
            for ri in ring_inds:
                lidar0_data[:,ring0[:]==ri] = 0

        # print('lidar_data', lidar_data.shape)

        radar_data = np.transpose(radar_data)
        V = 700*self.nsweeps
        if radar_data.shape[0] > V:
            print('radar_data', radar_data.shape)
            print('max pts', V)
            assert(False) # i expect this to never happen
            radar_data = radar_data[:V]
        elif radar_data.shape[0] < V:
            radar_data = np.pad(radar_data,[(0,V-radar_data.shape[0]),(0,0)],mode='constant')
        radar_data = np.transpose(radar_data)

        lidar0_data = torch.from_numpy(lidar0_data)
        lidar0_extra = torch.from_numpy(lidar0_extra)
        lidar_data = torch.from_numpy(lidar_data)
        lidar_extra = torch.from_numpy(lidar_extra)
        radar_data = torch.from_numpy(radar_data)

        binimg = (binimg > 0).float()
        seg_bev = (seg_bev > 0).float()
        
        # if self.include_extra:
        if self.get_tids:
            return imgs, rots, trans, intrins, lidar0_data, lidar0_extra, lidar_data, lidar_extra, lrtlist, vislist, tidlist_, scorelist, seg_bev, valid_bev, center_bev, offset_bev, size_bev, ry_bev, ycoord_bev, radar_data, egopose
        else:
            return imgs, rots, trans, intrins, lidar0_data, lidar0_extra, lidar_data, lidar_extra, lrtlist, vislist, scorelist, seg_bev, valid_bev, center_bev, offset_bev, size_bev, ry_bev, ycoord_bev, radar_data, egopose
    
    def __getitem__(self, index):


        cams = self.choose_cams()
        #if self.seqlen==1:
        #    return self.get_single_item(self.indices[index][0], cams)

        if self.is_train:
            # randomly sample the ref cam
            cam_id = np.random.randint(1, len(cams))
        else:
            cam_id = self.cam_id
        # print('seq index', index)
        # print('choosing refcam', cam_id)
        
        all_imgs = []
        all_rots = []
        all_trans = []
        all_intrins = []
        all_lidar0_data = []
        all_lidar0_extra = []
        all_lidar_data = []
        all_lidar_extra = []
        all_lrtlist = []
        all_vislist = []
        all_tidlist = []
        all_scorelist = []
        all_seg_bev = []
        all_valid_bev = []
        all_center_bev = []
        all_offset_bev = []
        all_radar_data = []
        all_egopose = []
        for index_t in self.indices[index]:
            # print('grabbing index %d' % index_t)
            imgs, rots, trans, intrins, lidar0_data, lidar0_extra, lidar_data, lidar_extra, lrtlist, vislist, tidlist, scorelist, seg_bev, valid_bev, center_bev, offset_bev, size_bev, ry_bev, ycoord_bev, radar_data, egopose = self.get_single_item(index_t, cams, cam_id=cam_id)

            # print('rots', rots)
            
            all_imgs.append(imgs)
            all_rots.append(rots)
            all_trans.append(trans)
            all_intrins.append(intrins)
            all_lidar0_data.append(lidar0_data)
            all_lidar0_extra.append(lidar0_extra)
            all_lidar_data.append(lidar_data)
            all_lidar_extra.append(lidar_extra)
            all_lrtlist.append(lrtlist)
            all_vislist.append(vislist)
            all_tidlist.append(tidlist)
            all_scorelist.append(scorelist)
            all_seg_bev.append(seg_bev)
            all_valid_bev.append(valid_bev)
            all_center_bev.append(center_bev)
            all_offset_bev.append(offset_bev)
            all_radar_data.append(radar_data)
            all_egopose.append(egopose)

        all_imgs = torch.stack(all_imgs)
        all_rots = torch.stack(all_rots)
        all_trans = torch.stack(all_trans)
        all_intrins = torch.stack(all_intrins)
        all_lidar0_data = torch.stack(all_lidar0_data)
        all_lidar0_extra = torch.stack(all_lidar0_extra)
        all_lidar_data = torch.stack(all_lidar_data)
        all_lidar_extra = torch.stack(all_lidar_extra)
        all_lrtlist = torch.stack(all_lrtlist)
        all_vislist = torch.stack(all_vislist)
        # all_tidlist = torch.stack(all_tidlist)
        all_scorelist = torch.stack(all_scorelist)
        all_seg_bev = torch.stack(all_seg_bev)
        all_valid_bev = torch.stack(all_valid_bev)
        all_center_bev = torch.stack(all_center_bev)
        all_offset_bev = torch.stack(all_offset_bev)
        all_radar_data = torch.stack(all_radar_data)
        all_egopose = torch.stack(all_egopose)

        # # i want to come up with coherent unique tids
        # print('all_tidlist', all_tidlist)
        # import ipdb; ipdb.set_trace()
        
        usable_tidlist = -1*torch.ones_like(all_scorelist).long()
        counter = 0
        for t in range(len(all_tidlist)):
            for i in range(len(all_tidlist[t])):
                if t==0:
                    usable_tidlist[t,i] = counter
                    counter += 1
                else:
                    st = all_tidlist[t][i]
                    if st in all_tidlist[0]:
                        usable_tidlist[t,i] = all_tidlist[0].index(st)
                    else:
                        usable_tidlist[t,i] = counter
                        counter += 1
        all_tidlist = usable_tidlist



        # maybe here i can also collect the object gt, as in:
        # clists that are valid on frame0, and continue to the end of the video

        # traj_xyz

        # collect_object_info(lrtlist_camRs, tidlist_s, scorelist_s, K, boxlist_camRs=None, mod='', summ_writer=None)
        # obj_lrt_traj, obj_tid_traj, obj_score_traj = utils.misc.collect_object_info(
        #     __u0(lrtlist_cam0), __u0(tidlist), __u0(scorelist), K*2)
        


        return all_imgs, all_rots, all_trans, all_intrins, all_lidar0_data, all_lidar0_extra, all_lidar_data, all_lidar_extra, all_lrtlist, all_vislist, all_tidlist, all_scorelist, all_seg_bev, all_valid_bev, all_center_bev, all_offset_bev, all_radar_data, all_egopose

class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg, _ = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg

def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name, shuffle=True, nsweeps=1, nworkers_val=1, include_extra=True, drop_rings=False, seqlen=1, cam_id=1, get_tids=False,
                 include_rgbs=True, temporal_aug=False):
    print('loading nuscenes...')
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    print('making parser...')
    traindata = parser(nusc,
                       is_train=True,
                       data_aug_conf=data_aug_conf,
                       nsweeps=nsweeps,
                       grid_conf=grid_conf,
                       include_extra=include_extra,
                       drop_rings=drop_rings,
                       seqlen=seqlen,
                       cam_id=cam_id,
                       include_rgbs=include_rgbs,
                       get_tids=get_tids,
                       temporal_aug=temporal_aug)
    valdata = parser(nusc,
                     is_train=False,
                     data_aug_conf=data_aug_conf,
                     nsweeps=nsweeps,
                     grid_conf=grid_conf,
                     include_extra=include_extra,
                     seqlen=seqlen,
                     cam_id=cam_id,
                     include_rgbs=include_rgbs,
                     get_tids=get_tids,
                     temporal_aug=False)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=shuffle,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=shuffle,
                                            num_workers=nworkers_val,
                                            drop_last=True)
    print('data ready')
    return trainloader, valloader
