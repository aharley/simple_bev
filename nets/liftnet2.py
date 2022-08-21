import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

import utils.geom
import utils.vox
import utils.misc
import utils.basic

from torchvision.models.resnet import resnet18
from efficientnet_pytorch import EfficientNet

EPS = 1e-4

from functools import partial

class VoxelsSumming(torch.autograd.Function):
    """Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/tools.py#L193"""
    @staticmethod
    def forward(ctx, x, geometry, ranks):
        """The features `x` and `geometry` are ranked by voxel positions."""
        # Cumulative sum of all features.
        x = x.cumsum(0)

        # Indicates the change of voxel.
        mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        mask[:-1] = ranks[1:] != ranks[:-1]

        x, geometry = x[mask], geometry[mask]
        # Calculate sum of features within a voxel.
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        ctx.save_for_backward(mask)
        ctx.mark_non_differentiable(geometry)

        return x, geometry

    @staticmethod
    def backward(ctx, grad_x, grad_geometry):
        (mask,) = ctx.saved_tensors
        # Since the operation is summing, we simply need to send gradient
        # to all elements that were part of the summation process.
        indices = torch.cumsum(mask, 0)
        indices[mask] -= 1

        output_grad = grad_x[indices]

        return output_grad, None, None


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip

class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) 
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None

        return {
            'raw_feat': x,
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }

import torchvision
class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x

class Liftnet(nn.Module):
    def __init__(self, Z, Y, X, ZMAX,
                 use_radar=False,
                 use_lidar=False,
                 use_metaradar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_type="res101"):
        super(Liftnet, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])

        self.Z, self.Y, self.X = Z, Y, X
        self.D = Z//2
        self.ZMAX = ZMAX
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress   
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()
        
        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        # self.feat2d_dim = feat2d_dim = latent_dim+Z
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim + self.D)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim + self.D)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim + self.D, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim + self.D, version='b4')

        # self.downsample = self.encoder.downsample

        # BEV compressor
        if self.use_radar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y + 16*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y+1, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
        elif self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y+Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                # use simple sum
                pass

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            
        # set_bn_momentum(self, 0.1)

    def splat_to_bev(self, feats, coords_mem, Z, Y, X):
        """ Adapted from https://github.com/wayveai/fiery/blob/master/fiery/models/fiery.py#L222"""
        B,S,C,D,H,W = feats.shape
        output = torch.zeros((B,C,Z,X), dtype=torch.float, device=feats.device)
        output_ones = torch.zeros((B,1,Z,X), dtype=torch.float, device=feats.device)

        feats = feats.permute(0,1,3,4,5,2) # put channels on end

        # print('feats', feats.shape)
        # print('coords_mem', coords_mem.shape)

        N = S * D * H * W # number of 3D coordinates per batch
        for b in range(B):
            # flatten x
            x_b = feats[b].reshape(N, C)

            # # Convert positions to integer indices
            # coords_mem_b = ((coords_mem[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            coords_mem_b = coords_mem[b].reshape(N, 3).long()

            # drop elements that are outside the considered spatial extent
            valid = (
                (coords_mem_b[:, 0] >= 0)
                & (coords_mem_b[:, 0] < X)
                & (coords_mem_b[:, 1] >= 0)
                & (coords_mem_b[:, 1] < Y)
                & (coords_mem_b[:, 2] >= 0)
                & (coords_mem_b[:, 2] < Z)
            )
            x_b = x_b[valid]
            coords_mem_b = coords_mem_b[valid]

            # sort the tensor contents
            inds = (
                coords_mem_b[:, 2] * Y * X
                + coords_mem_b[:, 1] * X
                + coords_mem_b[:, 0]
            )
            sorting = inds.argsort()
            x_b, coords_mem_b, inds = x_b[sorting], coords_mem_b[sorting], inds[sorting]

            one_b = torch.ones_like(x_b[:,0:1])
            # project to BEV by summing within voxels
            # x_b, coords_mem_b = VoxelsSumming.apply(x_b, coords_mem_b, inds)

            # print('x_b0', x_b.shape)
            # print('one_b0', one_b.shape)
            
            x_b, _ = VoxelsSumming.apply(x_b, coords_mem_b.clone(), inds)
            one_b, coords_mem_b = VoxelsSumming.apply(one_b, coords_mem_b, inds)
            # print('x_b', x_b.shape)
            # print('one_b', one_b.shape)
            # print('x_b', x_b.shape)
            # print('coords_mem_b', coords_mem_b.shape)

            x_b = x_b / one_b.clamp(min=1.0)

            bev_feature = torch.zeros((Z,Y,X,C), device=x_b.device)
            bev_ones = torch.zeros((Z,Y,X,1), device=x_b.device)
            bev_feature[coords_mem_b[:,2],coords_mem_b[:,1],coords_mem_b[:,0]] = x_b # Z,Y,X,C
            bev_ones[coords_mem_b[:,2],coords_mem_b[:,1],coords_mem_b[:,0]] = one_b # Z,Y,X,C
            # print('bev_feature', bev_feature.shape)

            bev_feature = bev_feature.sum(dim=1) # Z,X,C
            bev_feature = bev_feature.permute(2,0,1) # C,Z,X

            bev_ones = bev_ones.sum(dim=1) # Z,X,C
            bev_ones = bev_ones.permute(2,0,1) # C,Z,X
            
            output[b] = bev_feature
            output_ones[b] = bev_ones
        output = output / output_ones.clamp(min=1)

        return output

    # def create_frustum(self):
    #     # make grid in image plane
    #     ogfH, ogfW = self.data_aug_conf['final_dim']
    #     fH, fW = ogfH // self.downsample, ogfW // self.downsample
    #     ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
    #     D, _, _ = ds.shape
    #     xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
    #     ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

    #     # D x H x W x 3
    #     frustum = torch.stack((xs, ys, ds), -1)
    #     return nn.Parameter(frustum, requires_grad=False)

    # def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
    #     """Determine the (x,y,z) locations (in the ego frame)
    #     of the points in the point cloud.
    #     Returns B x N x D x H/downsample x W/downsample x 3
    #     """
    #     B, N, _ = trans.shape

    #     # undo post-transformation
    #     # B x N x D x H x W x 3
    #     points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
    #     points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

    #     # cam_to_ego
    #     points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
    #                         points[:, :, :, :, :, 2:3]
    #                         ), 5)
    #     combine = rots.matmul(torch.inverse(intrins))
    #     points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    #     points += trans.view(B, N, 1, 1, 1, 3)

    #     return points

        
    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
        - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        '''
        B, S, C, H, W = rgb_camXs.shape
        assert(C==3)
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)

        # rgb encoder
        device = rgb_camXs_.device
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])
        feat_camXs_ = self.encoder(rgb_camXs_)
        if self.rand_flip:
            feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, CD, Hf, Wf = feat_camXs_.shape

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X
        assert(CD==(self.D+self.feat2d_dim))

        # print('self.D', self.D)
        
        depth_camXs_ = feat_camXs_[:,:self.D].unsqueeze(1) # BS,1,D,Hf,Wf
        # depth_camXs_ = torch.randn_like(depth_camXs_)
        feat_camXs_ = feat_camXs_[:,self.D:].unsqueeze(2) # BS,C,1,Hf,Wf

        # feat_camXs_ = torch.ones_like(feat_camXs_)
        # # depth_camXs_ = (depth_camXs_/0.07).softmax(dim=2) # BS,1,D,Hf,Wf
        depth_camXs_ = (depth_camXs_).softmax(dim=2) # BS,1,D,Hf,Wf
        # utils.basic.print_stats('feat_camXs_', feat_camXs_)

        # depth_camXs_ = depth_camXs_ * 0.0
        # depth_camXs_[:,:,int(self.D/2)] = 1.0
        # depth_camXs_[:,:,0] = 1.0
        # depth_camXs_[:,:,-1] = 1.0
        # utils.basic.print_stats('feat_camXs_', feat_camXs_)
        
        # depth_camXs_ = (depth_camXs_/0.1).softmax(dim=2) # BS,1,D,Hf,Wf
        # depth_camXs_ = depth_camXs_.mean(dim=2, keepdim=True).repeat(1,1,Z,1,1) # uniform 
        feat_tileXs_ = feat_camXs_ * depth_camXs_ # BS,C,D,Hf,Wf
        feat_tileXs = __u(feat_tileXs_) # B,S,C,D,Hf,Wf
        # z_tileB = (D-1.0) * (z_camB-float(DMIN)) / float(DMAX-DMIN)
        # utils.basic.print_stats('feat_tileXs', feat_tileXs)

        # xyz_pixXs = utils.geom.meshgrid3d(B, self.D, Hf, Wf)
        # xyz_pixXs[:,:,2
        xyd_pixXs_ = utils.basic.gridcloud3d(B*S, self.D, Hf, Wf) # BS,DHW,3
        DMIN = 2.0 # slightly ahead of the cam
        DMAX = int(np.sqrt(self.ZMAX**2 + self.ZMAX**2)*0.9)
        xyd_pixXs_[:,:,2] = (xyd_pixXs_[:,:,2]/(self.D-1) * (DMAX-DMIN)) + DMIN # put into range [DMIN,DMAX]
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        # xyz_camXs_ = utils.geom.xyd2pointcloud(xyd_pixXs_, __p(pix_T_cams))
        xyz_camXs_ = utils.geom.xyd2pointcloud(xyd_pixXs_, featpix_T_cams_)
        xyz_cam0s_ = utils.geom.apply_4x4(__p(cam0_T_camXs), xyz_camXs_)
        xyz_mem0s_ = vox_util.Ref2Mem(xyz_cam0s_, Z, Y, X, assert_cube=False)
        xyz_mem0s = __u(xyz_mem0s_) # B,S,DHW,3
        # xyz_mem0s[:,:,:,1] = 1 # set all Y coords the same 
        # xyz_mem0s[:,:,:,1] = 0 # set all Y coords to 0
        feat_bev = self.splat_to_bev(feat_tileXs, xyz_mem0s, Z, Y, X)
        # print('feat_bev', feat_bev.shape)
        # utils.basic.print_stats('feat_bev', feat_bev)
        
        # # unproject image feature to 3d grid
        # featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        # feat_mems_ = vox_util.warp_tiled_to_mem(
        #     feat_tileXs_,
        #     utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
        #     camXs_T_cam0_, Z, Y, X, 5.0, self.ZMAX+5.0)
        # feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X

        # one_mems_ = vox_util.warp_tiled_to_mem(
        #     torch.ones_like(feat_tileXs_),
        #     utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
        #     camXs_T_cam0_, Z, Y, X, 5.0, self.ZMAX+5.0)
        # one_mems = __u(one_mems_) # B, S, C, Z, Y, X

        # one_mems = one_mems.clamp(min=1.0)
        # # mask_mems = (torch.abs(feat_mems) > 0).float()
        # # feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X
        # feat_mem = utils.basic.reduce_masked_mean(feat_mems, one_mems, dim=1) # B, C, Z, Y, X
        

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_bev[self.bev_flip1_index] = torch.flip(feat_bev[self.bev_flip1_index], [-1])
            feat_bev[self.bev_flip2_index] = torch.flip(feat_bev[self.bev_flip2_index], [-2])

        #     if rad_occ_mem0 is not None:
        #         rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
        #         rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        # bev compressing
        assert(not self.use_radar)
        assert(not self.use_lidar)
        assert(not self.do_rgbcompress)
        # feat_bev = torch.sum(feat_mem, dim=3)

        # bev decoder
        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation']
        center_e = out_dict['instance_center']
        offset_e = out_dict['instance_offset']

        return raw_e, feat_bev, seg_e, center_e, offset_e

