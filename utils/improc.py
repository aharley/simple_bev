import numpy as np
import torch
import cv2

import utils.basic

# color conversion libs, for flow vis
from skimage.color import (
    rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
    rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb)

def _convert(input_, type_):
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)

def _generic_transform_sk_3d(transform, in_type='', out_type=''):
    def apply_transform_individual(input_):
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        input_ = input_.permute(1, 2, 0).detach().numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(2, 0, 1)
        output = _convert(output, out_type)
        return output.to(device)

    def apply_transform(input_):
        to_stack = []
        for image in input_:
            to_stack.append(apply_transform_individual(image))
        return torch.stack(to_stack)
    return apply_transform

hsv_to_rgb = _generic_transform_sk_3d(hsv2rgb)

def back2color(i, blacken_zeros=False):
    if blacken_zeros:
        const = torch.tensor([-0.5])
        i = torch.where(i==0.0, const.cuda() if i.is_cuda else const, i)
        return back2color(i)
    else:
        return ((i+0.5)*255).type(torch.ByteTensor)

def colorize(d):
    if d.ndim==2:
        d = d.unsqueeze(dim=0)
    else:
        assert(d.ndim==3)
    d = d.repeat(3, 1, 1)
    return d

def oned2inferno(d, norm=True):
    # convert a 1chan input to a 3chan image output

    # if it's just B x H x W, add a C dim
    if d.ndim==3:
        d = d.unsqueeze(dim=1)
    # d should be B x C x H x W, where C=1
    B, C, H, W = list(d.shape)
    assert(C==1)

    if norm:
        d = utils.basic.normalize(d)
        
    rgb = torch.zeros(B, 3, H, W)
    for b in list(range(B)):
        rgb[b] = colorize(d[b])

    rgb = (255.0*rgb).type(torch.ByteTensor)

    return rgb

def convert_occ_to_height(occ, reduce_axis=3):
    B, C, D, H, W = list(occ.shape)
    assert(C==1)
    # note that height increases DOWNWARD in the tensor
    # (like pixel/camera coordinates)
    
    G = list(occ.shape)[reduce_axis]
    values = torch.linspace(float(G), 1.0, steps=G, dtype=torch.float32, device=occ.device)
    if reduce_axis==2:
        # fro view
        values = values.view(1, 1, G, 1, 1)
    elif reduce_axis==3:
        # top view
        values = values.view(1, 1, 1, G, 1)
    elif reduce_axis==4:
        # lateral view
        values = values.view(1, 1, 1, 1, G)
    else:
        assert(False) # you have to reduce one of the spatial dims (2-4)
    values = torch.max(occ*values, dim=reduce_axis)[0]/float(G)
    # values = values.view([B, C, D, W])
    return values

def draw_frame_id_on_vis(vis, frame_id, scale=0.5, left=5, top=20):

    rgb = vis.detach().cpu().numpy()[0]
    rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = (255, 255, 255)

    frame_str = utils.basic.strnum(frame_id)

    cv2.putText(
        rgb,
        frame_str,
        (left, top), # from left, from top
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, # font scale (float)
        color,
        1) # font thickness (int)
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    vis = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return vis

class Summ_writer(object):
    def __init__(self, writer, global_step, log_freq=10, fps=8, scalar_freq=100, just_gif=False):
        self.writer = writer
        self.global_step = global_step
        self.log_freq = log_freq
        self.fps = fps
        self.just_gif = just_gif
        self.maxwidth = 10000
        self.save_this = (self.global_step % self.log_freq == 0)
        self.scalar_freq = max(scalar_freq,1)

    def summ_gif(self, name, tensor, blacken_zeros=False):
        # tensor should be in B x S x C x H x W

        assert tensor.dtype in {torch.uint8,torch.float32}
        shape = list(tensor.shape)

        if tensor.dtype == torch.float32:
            tensor = back2color(tensor, blacken_zeros=blacken_zeros)

        video_to_write = tensor[0:1]

        S = video_to_write.shape[1]
        if S==1:
            # video_to_write is 1 x 1 x C x H x W
            self.writer.add_image(name, video_to_write[0,0], global_step=self.global_step)
        else:
            self.writer.add_video(name, video_to_write, fps=self.fps, global_step=self.global_step)

        return video_to_write

    def summ_scalar(self, name, value):
        if (not (isinstance(value, int) or isinstance(value, float) or isinstance(value, np.float32))) and ('torch' in value.type()):
            value = value.detach().cpu().numpy()
        if not np.isnan(value):
            if (self.log_freq == 1):
                self.writer.add_scalar(name, value, global_step=self.global_step)
            elif self.save_this or np.mod(self.global_step, self.scalar_freq)==0:
                self.writer.add_scalar(name, value, global_step=self.global_step)

    def summ_oned(self, name, im, bev=False, fro=False, logvis=False, max_val=0, max_along_y=False, norm=True, frame_id=None, only_return=False):
        if self.save_this:
            if bev:
                B, C, H, _, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=3)[0]
                else:
                    im = torch.mean(im, dim=3)
            elif fro:
                B, C, _, H, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=2)[0]
                else:
                    im = torch.mean(im, dim=2)
            else:
                B, C, H, W = list(im.shape)

            im = im[0:1] # just the first one
            assert(C==1)

            if logvis and max_val:
                max_val = np.log(max_val)
                im = torch.log(im)
                im = torch.clamp(im, 0, max_val)
                im = im/max_val
                norm = False
            elif max_val:
                im = torch.clamp(im, 0, max_val)/max_val
                norm = False

            vis = oned2inferno(im, norm=norm)
            if W > self.maxwidth:
                vis = vis[...,:self.maxwidth]
            return self.summ_rgb(name, vis, blacken_zeros=False, frame_id=frame_id, only_return=only_return)

    def summ_rgb(self, name, ims, blacken_zeros=False, frame_id=None, only_return=False, halfres=False):
        if self.save_this:
            assert ims.dtype in {torch.uint8,torch.float32}

            if ims.dtype == torch.float32:
                ims = back2color(ims, blacken_zeros)

            #ims is B x C x H x W
            vis = ims[0:1] # just the first one
            B, C, H, W = list(vis.shape)

            if halfres:
                vis = F.interpolate(vis, scale_factor=0.5)

            if frame_id is not None:
                vis = draw_frame_id_on_vis(vis, frame_id)

            if int(W) > self.maxwidth:
                vis = vis[:,:,:,:self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis.unsqueeze(1), blacken_zeros)

    def summ_occ(self, name, occ, reduce_axes=[3], bev=False, fro=False, pro=False, frame_id=None, only_return=False):
        if self.save_this:
            B, C, D, H, W = list(occ.shape)
            if bev:
                reduce_axes = [3]
            elif fro:
                reduce_axes = [2]
            elif pro:
                reduce_axes = [4]
            for reduce_axis in reduce_axes:
                height = convert_occ_to_height(occ, reduce_axis=reduce_axis)
                if reduce_axis == reduce_axes[-1]:
                    return self.summ_oned(name=('%s_ax%d' % (name, reduce_axis)), im=height, norm=False, frame_id=frame_id, only_return=only_return)
                else:
                    self.summ_oned(name=('%s_ax%d' % (name, reduce_axis)), im=height, norm=False, frame_id=frame_id, only_return=only_return)

    def flow2color(self, flow, clip=50.0):
        """
        :param flow: Optical flow tensor.
        :return: RGB image normalized between 0 and 1.
        """

        # flow is B x C x H x W

        B, C, H, W = list(flow.size())

        flow = flow.clone().detach()

        abs_image = torch.abs(flow)
        flow_mean = abs_image.mean(dim=[1,2,3])
        flow_std = abs_image.std(dim=[1,2,3])

        if clip:
            flow = torch.clamp(flow, -clip, clip)/clip
        else:
            # Apply some kind of normalization. Divide by the perceived maximum (mean + std*2)
            flow_max = flow_mean + flow_std*2 + 1e-10
            for b in range(B):
                flow[b] = flow[b].clamp(-flow_max[b].item(), flow_max[b].item()) / flow_max[b].clamp(min=1)

        radius = torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True)) #B x 1 x H x W
        radius_clipped = torch.clamp(radius, 0.0, 1.0)

        angle = torch.atan2(flow[:, 1:], flow[:, 0:1]) / np.pi #B x 1 x H x W

        hue = torch.clamp((angle + 1.0) / 2.0, 0.0, 1.0)
        saturation = torch.ones_like(hue) * 0.75
        value = radius_clipped
        hsv = torch.cat([hue, saturation, value], dim=1) #B x 3 x H x W

        #flow = tf.image.hsv_to_rgb(hsv)
        flow = hsv_to_rgb(hsv)
        flow = (flow*255.0).type(torch.ByteTensor)
        return flow

    def summ_flow(self, name, im, clip=0.0, only_return=False, frame_id=None):
        # flow is B x C x D x W
        if self.save_this:
            return self.summ_rgb(name, self.flow2color(im, clip=clip), only_return=only_return, frame_id=frame_id)
        else:
            return None
