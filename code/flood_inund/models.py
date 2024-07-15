__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 11/11/2019
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
__version__ = "1.6.0"

from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from decoder import ConvTransformerTokensToEmbeddingNeck
from encoder import MaskedAutoencoderViT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from unet_parts import *

import torch.nn as nn

from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        Instantiate a simplified U-Net network for image segmentation.
        :param n_channels: Number of input channels (e.g., 7)
        :param n_classes: Number of output classes (e.g., 3)
        """
        super(SimpleUNet, self).__init__()

        # Activation function
        self.relu = nn.ReLU()

        # Encoder (downsampling path)
        self.enc1 = self.conv_block(n_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)

        # Decoder (upsampling path)
        self.dec1 = self.conv_block(64, 32)
        self.dec2 = self.conv_block(32, 16)
        self.dec3 = nn.Conv2d(16, n_classes, kernel_size=1)


    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.relu,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.relu,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def upsample(self, x, target_size):
        return nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder (downsampling)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # Decoder (upsampling)
        dec1 = self.dec1(self.upsample(enc3, enc2.size()[2:]))
        dec2 = self.dec2(self.upsample(dec1, enc1.size()[2:]))
        dec3 = self.dec3(self.upsample(dec2, x.size()[2:]))

        return dec3



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,
                 ultrasmall=False,
                 device=torch.device('cuda')):
        """
        Instantiate a UNet network.
        :param n_channels: Number of input channels (e.g, 3 for RGB)
        :param n_classes: Number of output classes
        :param height: Height of the input images
        :param known_n_points: If you know the number of points,
                               (e.g, one pupil), then set it.
                               Otherwise it will be estimated by a lateral NN.
                               If provided, no lateral network will be build
                               and the resulting UNet will be a FCN.
        :param ultrasmall: If True, the 5 central layers are removed,
                           resulting in a much smaller UNet.
        :param device: Which torch device to use. Default: CUDA (GPU).
        """
        super(UNet, self).__init__()

        self.ultrasmall = ultrasmall
        self.device = device

        self.inc = inconv(n_channels, 8)
        self.down1 = down(8, 16)
        self.down2 = down(16, 32)
        if self.ultrasmall:
            self.down3 = down(32, 64, normaliz=False)
            self.up1 = up(96, 32)
            self.up2 = up(48, 16)
            self.up3 = up(24, 8, activ=False)
        else:
            self.down3 = down(16, 16)
            self.down4 = down(16, 16)
            self.down5 = down(16, 32)
            self.down6 = down(32, 32)
            self.down7 = down(32, 32, normaliz=False)

            self.up1 = up(64, 32)
            self.up2 = up(64, 16)
            self.up3 = up(32, 16)
            self.up4 = up(32, 16)
            self.up5 = up(32, 8)
            self.up6 = up(16, 8)
            self.up7 = up(16, 8, activ=False)

        self.outc = outconv(8, n_classes)
        self.out_nonlin = nn.Sigmoid()
        # self.out_nonlin = nn.ReLU()

        # This layer is not connected anywhere
        # It is only here for backward compatibility
        self.lin = nn.Linear(1, 1, bias=False)


    def forward(self, x):

        batch_size = x.shape[0]

        x1 = self.inc(x)
        # print("x1: ", x1.shape)
        x2 = self.down1(x1)
        # print("x2: ", x2.shape)
        x3 = self.down2(x2)
        # print("x3: ", x3.shape)
        x4 = self.down3(x3)
        # print("x4: ", x4.shape)
        if self.ultrasmall:
            x = self.up1(x4, x3)
            # print("x up1: ", x.shape)
            x = self.up2(x, x2)
            # print("x up2: ", x.shape)
            x = self.up3(x, x1)
            # print("x up3: ", x.shape)
        else:
            x5 = self.down4(x4)
            # print("x5: ", x5.shape)
            x6 = self.down5(x5)
            # print("x6: ", x6.shape)
            x7 = self.down6(x6)
            # print("x7: ", x7.shape)
            x8 = self.down7(x7)
            # print("x8: ", x8.shape)

            x = self.up1(x8, x7)
            # print("up1: ", x.shape)
            x = self.up2(x, x6)
            # print("up2: ", x.shape)
            x = self.up3(x, x5)
            # print("up3: ", x.shape)
            x = self.up4(x, x4)
            # print("up4: ", x.shape)
            x = self.up5(x, x3)
            # print("up5: ", x.shape)
            x = self.up6(x, x2)
            # print("up6: ", x.shape)
            x = self.up7(x, x1)
            # print("up7: ", x.shape)

        x = self.outc(x)
        x = self.out_nonlin(x)

        return x


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 11/11/2019
"""

# MODIFIED FROM https://github.com/isaaccorley/prithvi-pytorch
class PrithviEncoder(nn.Module):
    def __init__(
        self,
        cfg_path: str,
        ckpt_path: Optional[str] = None,
        num_frames: int = 1,
        in_chans: int = 6,
        img_size: int = 224,
    ):
        super().__init__()
        cfg = OmegaConf.load(cfg_path)
        cfg.model_args.num_frames = num_frames
        cfg.model_args.in_chans = in_chans
        cfg.model_args.img_size = img_size

        self.embed_dim = cfg.model_args.embed_dim
        self.depth = cfg.model_args.depth
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = cfg.model_args.patch_size
        encoder = MaskedAutoencoderViT(**cfg.model_args)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")

            if num_frames != 3:
                #del state_dict["pos_embed"]
                #del state_dict["decoder_pos_embed"]
                pass

            if in_chans != 6:
                #del state_dict["patch_embed.proj.weight"]
                #del state_dict["decoder_pred.weight"]
                #del state_dict["decoder_pred.bias"]
                pass

            encoder.load_state_dict(state_dict, strict=False)
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add a temporal dimension if num_frames = 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")

        x, _, _ = self.encoder.forward_encoder(x, mask_ratio=0.0)

        # Squeeze temporal dim if t=1
        x = x.squeeze(dim=2)
        return x

    def forward_features(
        self,
        x: torch.Tensor,
        n: list[int],
        mask_ratio: float = 0.0,
        reshape: bool = True,
        norm=False,
    ):
        # add a temporal dimension if num_frames = 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")

        x = self.encoder.get_intermediate_layers(
            x, n=n, mask_ratio=mask_ratio, reshape=reshape, norm=norm
        )
        return x

# MODIFIED FROM https://github.com/isaaccorley/prithvi-pytorch
class PrithviViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cfg_path: str,
        ckpt_path: Optional[str] = None,
        in_chans: int = 6,
        img_size: int = 224,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PrithviEncoder(
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            num_frames=1,
            in_chans=in_chans,
            img_size=img_size,
        )
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = nn.Linear(
            in_features=self.encoder.embed_dim, out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        #x = x[:, 0]  # cls token
        return self.head(x)

# MODIFIED FROM https://github.com/isaaccorley/prithvi-pytorch
class PrithviUnet(Unet):
    def __init__(
        self,
        num_classes: int,
        cfg_path: str,
        ckpt_path: Optional[str] = None,
        in_chans: int = 6,
        img_size: int = 224,
        n: list[int] = [2, 5, 8, 11],
        norm: bool = True,
        decoder_channels: list[int] = [256, 128, 64, 32],
        freeze_encoder: bool = False,
    ):
        super().__init__(encoder_weights=None)
        assert len(n) == 4, "Num intermediate blocks must be 5"
        self.n = n
        self.num_classes = num_classes
        self.norm = norm
        self.encoder = PrithviEncoder(
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            num_frames=1,
            in_chans=in_chans,
            img_size=img_size,
        )
        assert all(
            [i < self.encoder.depth for i in n]
        ), "intermediate block index must be less than the ViT depth"

        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self._depth = 4
        self._in_channels = in_chans
        self._out_channels = [in_chans] + [self.encoder.embed_dim] * len(self.n)
        self.upsample = nn.ModuleList(
            [nn.UpsamplingBilinear2d(scale_factor=s) for s in self.scale_factors]
        )

        self.decoder = UnetDecoder(
            encoder_channels=self._out_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )
        self.classification_head = None
        self.name = "u-prithvi"
        self.initialize()

    def forward(self, x):
        features = self.encoder.forward_features(
            x, n=self.n, mask_ratio=0.0, reshape=True, norm=self.norm
        )
        features = [up(f) for up, f in zip(self.upsample, features)]
        features = [x] + list(features)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks

    @property
    def scale_factors(self):
        num_tokens = self.encoder.img_size // self.encoder.patch_size
        sizes = [
            self.encoder.img_size // i
            for i in [2 ** (i + 1) for i in range(len(self.n))]
        ]
        return [s // num_tokens for s in sizes]

# MODIFIED FROM https://github.com/isaaccorley/prithvi-pytorch
class PrithviEncoderDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cfg_path: str,
        ckpt_path: Optional[str] = None,
        in_chans: int = 6,
        img_size: int = 224,
        freeze_encoder: bool = False,
        num_neck_filters: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PrithviEncoder(
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            num_frames=1,
            in_chans=in_chans,
            img_size=img_size,
        )
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        num_tokens = self.encoder.img_size // self.encoder.patch_size
        self.decoder = ConvTransformerTokensToEmbeddingNeck(
            embed_dim=self.encoder.embed_dim,
            output_embed_dim=num_neck_filters,
            Hp=num_tokens,
            Wp=num_tokens,
            drop_cls_token=True,
        )
        self.head = nn.Conv2d(
            in_channels=num_neck_filters,
            out_channels=num_classes,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x
    


# BELOW CODE MODIFIED FROM https://github.com/lucidrains/segformer-pytorch
# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        #print(x.shape) # should have 6 channels
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)


            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)
            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

class Segformer(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 4
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x):
    
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]

        fused = torch.cat(fused, dim = 1)
        
        fused = self.to_segmentation(fused)

        fused = F.interpolate(fused, scale_factor=4, mode='bilinear', align_corners=True)
        
        return fused
    
