import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import ModelRegistry
from ..heads import BCEHead 


def drop_path_func(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_func(x, self.drop_prob, self.training)


def window_partition(x, window_size):
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, B, L):
    C = windows.shape[-1]
    x = windows.view(B, L // window_size, window_size, C)
    x = x.contiguous().view(B, L, C)
    return x


class WindowAttention1d(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads))

        coords = torch.arange(self.window_size)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += self.window_size - 1
        self.register_buffer("relative_position_index", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock1d(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = self.input_resolution

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention1d(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):
        L = self.input_resolution
        B, L_in, C = x.shape
        assert L == L_in

        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows)
        shifted_x = window_reverse(attn_windows, self.window_size, B, L)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging1d(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        B, L, C = x.shape
        assert L % 2 == 0

        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]
        x = torch.cat([x0, x1], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed1d(nn.Module):
    def __init__(self, seq_len=5000, patch_size=4, in_chans=12, embed_dim=96, norm_layer=None):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformer1d(nn.Module):
    def __init__(self, model_config, seq_len=5000, patch_size=4, in_chans=12,
                 embed_dim=96, depths=[2, 2], num_heads=[3, 6],
                 window_sizes=[50, 625], shift_sizes=[25, 0],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed1d(
            seq_len=seq_len, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.input_resolution = num_patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.layers = nn.ModuleList()
        curr_res = self.input_resolution
        curr_dim = embed_dim
        dpr_idx = 0

        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            stage_blocks = nn.ModuleList([
                SwinTransformerBlock1d(
                    dim=curr_dim, input_resolution=curr_res,
                    num_heads=num_heads[i_layer], window_size=window_sizes[i_layer],
                    shift_size=0 if (j % 2 == 0) else shift_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[dpr_idx + j],
                    norm_layer=norm_layer)
                for j in range(depths[i_layer])
            ])
            dpr_idx += depths[i_layer]
            layer.append(stage_blocks)

            if i_layer < self.num_layers - 1:
                layer.append(PatchMerging1d(curr_res, curr_dim, norm_layer=norm_layer))
                curr_dim = curr_dim * 2
                curr_res = curr_res // 2

            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        num_classes = getattr(model_config, 'num_classes', 23)
        self.head = BCEHead(num_classes, self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            blocks = layer[0]
            for block in blocks:
                x = block(x)
            if len(layer) > 1:
                x = layer[1](x)

        x = self.norm(x)
        return self.head(x)


@ModelRegistry.register('swin1d_ecg')
def get_swin1d_ecg(model_config):
    return SwinTransformer1d(
        model_config,
        seq_len=5000,
        patch_size=4,
        in_chans=12,
        embed_dim=64,
        depths=[4, 2],
        num_heads=[4, 8],
        window_sizes=[50, 625],
        shift_sizes=[25, 0],
        drop_path_rate=0.1, 
    )
