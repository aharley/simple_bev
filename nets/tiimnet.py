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

import math
import copy
from torch import Tensor
from typing import Optional
from nets.mocha import MoChA
from torch.utils.checkpoint import checkpoint

from functools import partial
from einops.layers.torch import Rearrange, Reduce

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # Intermediate representations for Auxiliary losses
        self.return_intermediate = return_intermediate

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2, src2_weights = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if self.return_intermediate:
            # Want the output and attention map of the last layer
            return src, src2_weights
        else:
            return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, src2_weights = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        if self.return_intermediate:
            # Want the output and attention map of the last layer
            return src, src2_weights
        else:
            return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        # Intermediate representations for Auxiliary losses
        self.return_intermediate = return_intermediate

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src
        intermediate = None

        if self.return_intermediate:
            for layer in self.layers:
                output, intermediate = layer(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos=pos,
                )
        else:
            for layer in self.layers:
                output = layer(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos=pos,
                )

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            # Want the output and attention map of the last layer
            return output, intermediate
        else:
            return output

class TransformerDecoderLayerMMA(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.src_attn = MoChA(
            kdim=d_model,
            qdim=d_model,
            adim=d_model,
            odim=d_model,
            atype="scaled_dot",
            chunk_size=-1,
            n_heads_mono=nhead,
            n_heads_chunk=nhead,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, _, _, _ = self.src_attn(
            key=self.with_pos_embed(memory, pos).permute(1, 0, 2),
            value=memory.permute(1, 0, 2),
            query=self.with_pos_embed(tgt, query_pos).permute(1, 0, 2),
            mode="parallel",
        )
        tgt2 = tgt2.permute(1, 0, 2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, _, _, _ = self.src_attn(
            key=self.with_pos_embed(memory, pos),
            value=memory,
            query=self.with_pos_embed(tgt, query_pos),
            mode="parallel",
        )
        tgt2 = tgt2.permute(1, 0, 2)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt
        intermediate = None

        if self.return_intermediate:
            for layer in self.layers:
                output, intermediate = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    pos=pos,
                    query_pos=query_pos,
                )
        else:
            for layer in self.layers:
                output = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    pos=pos,
                    query_pos=query_pos,
                )

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            # Want the output and attention map of the last layer
            return output, intermediate
        else:
            return output

class TransformerMonotonic(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None 
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayerMMA(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, query_embed, pos_embed, tgt_mask=None):
        # flatten NxCxHxW to HWxNxC
        # bs, c, h, w = src.shape
        # src = src.flatten(2).permute(2, 0, 1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs

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

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) # note [-2] instead of [-3], since Y is gone now
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

class VanillaSelfAttention(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super(VanillaSelfAttention, self).__init__()
        self.dim = dim 
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=4, n_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, query_pos=None):
        '''
        query: (B, N, C)
        '''
        inp_residual = query

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        device = query.device
        Z, X = 200, 200
        ref_z, ref_x = torch.meshgrid(
            torch.linspace(0.5, Z-0.5, Z, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X-0.5, X, dtype=torch.float, device=query.device)
        )
        ref_z = ref_z.reshape(-1)[None] / Z
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_z, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2) # (B, N, 1, 2)

        B, N, C = query.shape
        input_spatial_shapes = query.new_zeros([1,2]).long()
        input_spatial_shapes[:] = 200
        input_level_start_index = query.new_zeros([1,]).long()
        queries = self.deformable_attention(query, reference_points, query.clone(), 
            input_spatial_shapes, input_level_start_index)

        queries = self.output_proj(queries)

        return self.dropout(queries) + inp_residual

class SpatialCrossAttention(nn.Module):
    # From https://github.com/zhiqi-li/BEVFormer

    def __init__(self, dim=128, dropout=0.1):
        super(SpatialCrossAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn3D(embed_dims=dim, num_heads=4, num_levels=1, num_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, query_pos=None, reference_points_cam=None, spatial_shapes=None, bev_mask=None):
        '''
        query: (B, N, C)
        key: (S, M, B, C)
        reference_points_cam: (S, B, N, D, 2), in 0-1
        bev_mask: (S. B, N, D)
        '''
        inp_residual = query
        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        S, M, _, _ = key.shape

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros(
            [B, S, max_len, self.dim])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [B, S, max_len, D, 2])

        for j in range(B):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        key = key.permute(2, 0, 1, 3).reshape(
            B * S, M, C)
        value = value.permute(2, 0, 1, 3).reshape(
            B * S, M, C)

        level_start_index = query.new_zeros([1,]).long()
        queries = self.deformable_attention(query=queries_rebatch.view(B*S, max_len, self.dim),
            key=key, value=value,
            reference_points=reference_points_rebatch.view(B*S, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index).view(B, S, max_len, self.dim)

        for j in range(B):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0 
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual

# no radar/lidar integration
class Tiimnet(nn.Module):
    def __init__(self, Z, Y, X, 
                 ZMAX=50,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_type="res101"):
        super(Tiimnet, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])

        self.Z, self.Y, self.X = Z, Y, X 
        self.ZMAX = ZMAX 
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.use_radar = False
        self.use_lidar = False

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()
        
        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # TIIM transformer
        self.pos_enc = PositionalEncoding(latent_dim, 0.1, 1000)
        self.query_embed = nn.Embedding(100, latent_dim)
        self.tbev8 = TransformerMonotonic(
            d_model = latent_dim,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
        )

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

    def trans_reshape(self, input):
        N, C, H, W = input.shape

        # [N, C, H, W] ----> [H, NW, C]
        # [N, C, H, W] ---> [H, C, N, W] ---> [H, C, NW] ---> [H, NW, C]
        x = input.permute(2, 1, 0, 3).flatten(2).permute(0, 2, 1)
        return x

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

    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        '''
        B, S, C, H, W = rgb_camXs.shape 
        B0 = B*S
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
        feat_camXs_ = self.encoder(rgb_camXs_) # (B0, C, Hf, Wf)
        if self.rand_flip:
            feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_camXs_.shape
        # feat_camXs = __u(feat_camXs_) # (B, S, C, Hf, Wf)

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        # translating images into maps
        DMIN = 2.0
        DMAX = int(np.sqrt(self.ZMAX**2 + self.ZMAX**2)*0.9)
        self.D = Z//2

        # feat_camXs_ B0, C, Hf, Wf
        tgt8 = torch.zeros_like(feat_camXs_[:, 0, :1]).expand(-1, self.D, -1)
        qe8 = (self.query_embed(tgt8.long())).permute(0, 3, 1, 2)
        tgt8 = (tgt8.unsqueeze(-1)).permute(0, 3, 1, 2)

        bev8 = checkpoint(
            self.tbev8,
            self.trans_reshape(feat_camXs_),
            self.pos_enc(self.trans_reshape(tgt8)),
            self.trans_reshape(qe8),
            self.pos_enc(self.trans_reshape(feat_camXs_)),
        ) # Z, B0*Wf, C

        xyd_pixXs_ = utils.basic.gridcloud3d(B*S, self.D, 1, Wf) # BS, DW, 3
        xyd_pixXs_[:,:,2] = (xyd_pixXs_[:,:,2]/(self.D-1) * (DMAX-DMIN)) + DMIN # put into range [DMIN,DMAX]
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        xyz_camXs_ = utils.geom.xyd2pointcloud(xyd_pixXs_, featpix_T_cams_)
        xyz_cam0s_ = utils.geom.apply_4x4(__p(cam0_T_camXs), xyz_camXs_)
        xyz_mem0s_ = vox_util.Ref2Mem(xyz_cam0s_, Z, Y, X, assert_cube=False)
        xyz_mem0s = __u(xyz_mem0s_) # B,S,DW,3
        # TODO: convert bev8 into feat_tileXs
        feat_tileXs_ = bev8.reshape(self.D, B0, Wf, self.latent_dim).permute(1, 3, 0, 2) # (B0, C, D, Wf)
        feat_tileXs = __u(feat_tileXs_).unsqueeze(-2) # B,S,C,D,1,Wf
        feat_bev = self.splat_to_bev(feat_tileXs, xyz_mem0s, Z, Y, X)

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_bev[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_bev[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

        # bev decoder
        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation']
        center_e = out_dict['instance_center']
        offset_e = out_dict['instance_offset']

        return raw_e, feat_e, seg_e, center_e, offset_e
