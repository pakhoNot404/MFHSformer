from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from .commons import AttentionModule
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from copy import deepcopy
from torch import Tensor
from typing import Optional, Sequence, Tuple, Union
from mmseg.models.decode_heads.uper_head import UPerHead


def project_vk_linformer(v, k, E):
    # project k,v
    v = torch.einsum('b h j d , j k -> b h k d', v, E)
    k = torch.einsum('b h j d , j k -> b h k d', k, E)
    return v, k


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        out = x.flatten(2).transpose(1, 2)
        out = self.norm(out)
        return out, H, W


class DWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(dim, dim // 2, kernel_size, 1, kernel_size // 2, groups=dim // 2)

    def forward(self, x):
        return self.conv(x)


class MSLAttention(nn.Module):
    def __init__(self,
                 dim, heads=8,
                 shared_projection=True, proj_shape=None, trainable_proj=True,
                 proj=True, attn_drop_ratio=0.,
                 ):
        super(MSLAttention, self).__init__()
        self.dim_head = (int(dim / heads))
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.shared_projection = shared_projection
        self.cnn_atten = AttentionModule(dim=dim)

        self.E = torch.nn.Parameter(torch.randn(proj_shape), requires_grad=trainable_proj)
        self.k = proj_shape[1]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.downsample = PatchMerging(dim=dim, norm_layer=norm_layer)
        self.proj = proj
        self.drop_path = DropPath(0.3)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

    def forward(self, x, proj_mat=None):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        assert x.dim() == 3

        if self.proj:
            E = self.E
            assert x.shape[1] == E.shape[0], f'{x.shape[1]} Token in the input sequence while' \
                                             f' {E.shape[0]} were provided in the E proj matrix'

        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # k: [2,12,3136,64]  v: [2,12,3136,64]  # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))
        qkv = qkv.reshape(B, L, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 以下代码是加入cnn attention
        cnn_attn = self.cnn_atten(x, H, W)
        v = rearrange(v, "b h i d -> b i (h d)")
        v = cnn_attn * v
        v = rearrange(v, "b i (h d) -> b h i d", h=self.heads)

        if self.proj:
            v, k = project_vk_linformer(v, k, E)  # [batch_size, num_heads, proj_shape[1], d]

        attn = (q @ k.transpose(-2, -1)) * self.scale_factor
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        out = self.W_0(x)

        return out


class MSLFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, feedforward_channels, proj_shape=None, drop_rate=0., drop_path_rate=0.,
                 drop_path=0., proj=True, flex=True, flex_k=4,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'), norm_layer=nn.LayerNorm, ):
        super().__init__()
        self.dim = dim // flex_k
        self.num_heads = num_heads
        self.flex_k = flex_k,
        self.norm1 = norm_layer(dim)
        self.attn = MSLAttention(
            dim=dim,
            heads=num_heads,
            shared_projection=None,
            proj_shape=proj_shape,
            proj=proj,
            trainable_proj=True, )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.ffn = FFN(
            embed_dims=dim,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

        self.flex = flex

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.resdwc = DWC(dim * 2)

    def _resize(self, x: Tensor, shape: Tuple[int, int]):
        x_resized = F.interpolate(
            x,
            shape,
            mode='bicubic',
            antialias=True,
        )
        return x_resized

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        self.attn.H, self.attn.W = H, W
        identity = x
        if self.flex:
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

            cnn_dw = x
            if self.flex_k[0] == 4:
                cnn_dw = self.conv1(x)
            elif self.flex_k[0] == 2:
                cnn_dw = self.conv2(x)

            newshape = (H // self.flex_k[0], W // self.flex_k[0])
            x = self._resize(x, newshape)
            x = rearrange(x, "b c h w -> b (h w) c", h=H // self.flex_k[0], w=W // self.flex_k[0])
            self.attn.H, self.attn.W = H // self.flex_k[0], W // self.flex_k[0]
            x = self.norm1(x)
            x = self.attn(x)
            x = self.drop_path(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=H // self.flex_k[0], w=W // self.flex_k[0])

            # concat
            cnn_dw = torch.cat([cnn_dw, x], dim=1)
            x = self.resdwc(cnn_dw)

            newshape = (H, W)
            x = self._resize(x, newshape)
            x = rearrange(x, "b c h w -> b (h w) c", h=H, w=W)


        else:
            x = self.norm1(x)
            x = self.attn(x)
            x = self.drop_path(x)

        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, feedforward_channels, proj_shape=True, drop_rate=0., flex=True, flex_k=4,
                 drop_path=0., drop_path_rate=0., norm_layer=nn.LayerNorm, downsample=None, proj=True, ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.proj_shape = proj_shape

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            MSLFormerBlock(
                dim=dim,
                num_heads=num_heads,
                proj=proj,
                flex=flex,
                flex_k=flex_k,
                proj_shape=proj_shape,
                drop_rate=drop_rate,
                feedforward_channels=feedforward_channels,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                drop_path_rate=drop_path_rates[i],
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x)
            hw_shape = (H, W)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            DH, DW = (H + 1) // 2, (W + 1) // 2
            return x_down, DH, DW, x, hw_shape
        else:
            return x, H, W, x, hw_shape


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class MSLFormer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_c=3,
                 num_classes=2,
                 embed_dim=96,
                 mlp_ratio=4,
                 depths=[3, 3, 9, 3],
                 num_heads=[4, 8, 16, 32],
                 out_indices=(0, 1, 2, 3),
                 flex_k=[4, 2, 2, 1],
                 proj_shape=[196, 196, 49, 24],
                 drop_ratio=0.,
                 drop_path_rate=0.3,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 norm_cfg=dict(type='LN'), ):

        super(MSLFormer, self).__init__()
        self.num_classes = num_classes
        self.out_indices = out_indices

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.num_layers = len(depths)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                drop_rate=drop_ratio,
                                drop_path_rate=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                feedforward_channels=int(mlp_ratio * embed_dim),
                                proj_shape=(proj_shape[i_layer], proj_shape[i_layer] // 4),
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                # proj=True,
                                flex=True if (i_layer < self.num_layers - 1) else False,
                                proj=True if (i_layer < self.num_layers - 1) else False,
                                flex_k=flex_k[i_layer],
                                )
            self.layers.append(layers)

        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)

    def forward_features(self, x):
        out_list = []
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x, H, W = self.patch_embed(x)  # [B, (H/4)*(W/4), embed_dim]
        # x = self.pos_drop(x + self.pos_embed)  # [B, (H/4)*(W/4), embed_dim] [B, 3136, 768]
        x = self.pos_drop(x)  # [B, (H/4)*(W/4), embed_dim] [B, 3136, 768]
        outs = []
        for i, layer in enumerate(self.layers):
            x, H, W, out, out_hw_shape = layer(x, H, W)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        return out, outs

    def forward(self, x):
        out, outs = self.forward_features(x)

        return out, outs


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class visiontrans(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MSLFormer(
            img_size=224,
            patch_size=4,
            embed_dim=128,
            depths=[3, 3, 9, 3],
            num_heads=[4, 8, 16, 32],  # [4, 8, 16, 32],
            flex_k=[4, 2, 2, 1],
            num_classes=2,
        )
        self.decode_head = UPerHead(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=256,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False
        )
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1, stride=1),
        )

    def forward(self, x):
        out, x = self.backbone(x)
        out = self.decode_head(x)  # [2,256,56,56]
        out = self.up_sample(out)
        out = F.interpolate(out,
                            size=[224, 224],
                            mode='bilinear')
        return out
