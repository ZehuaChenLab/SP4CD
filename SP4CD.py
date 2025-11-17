import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple
from .SimAM import SimAM
nonlinearity = nn.SiLU()
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import torch.nn as nn
import torch
import math
from einops import repeat


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        att_weight = self.activaton(y)
        return att_weight

class LN(nn.Module):
    def __init__(self, c_channel,norm_layer=nn.LayerNorm):
        super(LN, self).__init__()
        self.ln = nn.LayerNorm(c_channel)

    def forward(self,x):
            x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return x

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner*2 , bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        return y


class DAR(nn.Module):
    def __init__(self, in_channels):
        super(DAR, self).__init__()
        
        self.channel_H = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.channel_W = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        # self.spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
    def _swap_feature_blocks(self, feat1, feat2, num_splits=4, split_type='width'):
        
        B, C, H, W = feat1.shape
        if split_type == 'width':
            assert W % num_splits == 0, "Width must be divisible by num_splits"
        elif split_type == 'height':
            assert H % num_splits == 0, "Height must be divisible by num_splits"
        else:
            raise ValueError("Invalid split_type")
        
        time1 = []
        time2 = []

        if split_type == 'height':
            # Step 1: Split both features along height dimension
            split_size = H // num_splits
            feat1_splits = torch.split(feat1, split_size, dim=2)
            feat2_splits = torch.split(feat2, split_size, dim=2)
            
            for i in range(num_splits):
                x_1, x_2 = self.channel_attention(feat1_splits[i], feat2_splits[i], mode='height')
                time1.append(x_1)
                time2.append(x_2)
            x_1 = torch.cat(time1, dim=2)
            x_2 = torch.cat(time2, dim=2)
            
            x_final = x_1 + x_2
            
        elif split_type == 'width':
            # Step 1: Split both features along width dimension
            split_size = W // num_splits
            feat1_splits = torch.split(feat1, split_size, dim=3)  # list of 4 tensors
            feat2_splits = torch.split(feat2, split_size, dim=3)
            
            for i in range(num_splits):
                x_1, x_2 = self.channel_attention(feat1_splits[i], feat2_splits[i], mode='width')
                time1.append(x_1)
                time2.append(x_2)
            x_1 = torch.cat(time1, dim=3)
            x_2 = torch.cat(time2, dim=3)
            
            x_final = x_1 + x_2
            
        return x_final

    def channel_attention(self, x_1, x_2, mode='height'):
        
        if mode == 'height':
            x = torch.cat([x_1, x_2], dim=2)
            # x = torch.amax(x, dim=[2, 3], keepdim=True)
            x = torch.mean(x, dim=[2, 3], keepdim=True)
            
            x = self.channel_H(x)
            x_1 = F.sigmoid(x) * x_1
            x_2 = F.sigmoid(x) * x_2
            
        elif mode == 'width':
            x = torch.cat([x_1, x_2], dim=3)
            x = torch.mean(x, dim=[2, 3], keepdim=True)
            # x = torch.amax(x, dim=[2, 3], keepdim=True)
            x = self.channel_W(x)
            x_1 = F.sigmoid(x) * x_1
            x_2 = F.sigmoid(x) * x_2
                
        return x_1, x_2
            
    # def spatial_attention(self, x_1, x_2, mode='height'):
    #     if mode == 'height':
    #         x = torch.cat([x_1, x_2], dim=1)
    #         x_mean = torch.mean(x, dim=1, keepdim=True)
    #         x_max = torch.max(x, dim=1, keepdim=True)[0]
            
    #         x_spatial = self.spatial(torch.cat([x_mean, x_max], dim=1))
            
    #         x_1 = x_1 * F.sigmoid(x_spatial)
    #         x_2 = x_2 * F.sigmoid(x_spatial)
            
    #     elif mode == 'width':
    #         x = torch.cat([x_1, x_2], dim=1)
    #         x_mean = torch.mean(x, dim=1, keepdim=True)
    #         x_max = torch.max(x, dim=1, keepdim=True)[0]
            
    #         x_spatial = self.spatial(torch.cat([x_mean, x_max], dim=1))
            
    #         x_1 = x_1 * F.sigmoid(x_spatial)
    #         x_2 = x_2 * F.sigmoid(x_spatial)
            
    #     return x_1, x_2
    
    # def cbam_window(self, x_1, x_2, mode='height'):
    #     if mode == 'height':
    #         x_1, x_2 = self.channel_attention(x_1, x_2, mode='height')
    #         x_1, x_2 = self.spatial_attention(x_1, x_2, mode='height')
            
    #     elif mode == 'width':
    #         x_1, x_2 = self.channel_attention(x_1, x_2, mode='width')
    #         x_1, x_2 = self.spatial_attention(x_1, x_2, mode='width')
            
    #     return x_1, x_2

    def forward(self, x_1, x_2):
        assert x_1.shape == x_2.shape, "Input tensors must have the same shape"
        B, C, H, W = x_1.shape
        
        x_final_H = self._swap_feature_blocks(x_1, x_2, num_splits=4, split_type='height')
        x_final_W = self._swap_feature_blocks(x_1, x_2, num_splits=4, split_type='width')
        
        return x_final_H + x_final_W

def im2cswin(x, h_sp, w_sp):
    b, c, h, w = x.shape
    # b, c, h, w ==> b, c, h // h_sp, h_sp, w // w_sp, w_sp
    x = x.view(b, c, h // h_sp, h_sp, w // w_sp, w_sp)
    # b, c, h // h_sp, h_sp, w // w_sp, w_sp ==> b, h_sp, w_sp, c, h // h_sp, w // w_sp
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    # b, h_sp, w_sp, c, h // h_sp, w // w_sp ==> b * h_sp * w_sp, c, h // h_sp, w // w_sp
    x = x.view(-1, c, h // h_sp, w // w_sp)
    return x


def cswin2im(x, h_sp, w_sp, b):
    _, c, h, w = x.shape
    #  b * h_sp * w_sp, c, h // h_sp, w // w_sp ==> b, h_sp, w_sp, c, h // h_sp, w // w_sp
    x = x.view(b, h_sp, w_sp, c, h, w)
    # b, h_sp, w_sp, c, h // h_sp, w // w_sp ==> b, c, h // h_sp, h_sp, w // w_sp, w_sp
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    # b, h_sp, w_sp, c, h // h_sp, w // w_sp ==>
    x = x.view(b, c, h * h_sp, w * w_sp)
    return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



class LayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.layernorm = nn.LayerNorm(num_features, eps=eps)

    def forward(self, x):
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2)
        return x


class SP4CD_P(nn.Module):
    # 并行HW操作
    def __init__(self, in_channels, out_channels, h_sp, w_sp, mlp_ratio=4, nonlinearity=nn.SiLU()):
        
        super(SP4CD_P, self).__init__()
        self.h_sp = h_sp
        self.w_sp = w_sp

        # 时相1 H方向DW卷积+SS2D
        self.dw_h_1_ss2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=autopad(3),groups=in_channels),
            SS2D(d_model=in_channels//2),
            LayerNorm2d(in_channels))

        # 时相1 W方向DW卷积+SS2D
        self.dw_w_1_ss2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=autopad(3), groups=in_channels),
            SS2D(d_model=in_channels // 2),
            LayerNorm2d(in_channels))

        # 时相2 H方向DW卷积+SS2D
        self.dw_h_2_ss2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=autopad(3), groups=in_channels),
            SS2D(d_model=in_channels // 2),
            LayerNorm2d(in_channels))

        # 时相2 W方向DW卷积+SS2D
        self.dw_w_2_ss2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=autopad(3), groups=in_channels),
            SS2D(d_model=in_channels // 2),
            LayerNorm2d(in_channels))

        # 融合时相信息Mamba_Conv
        self.concat_conv = SimAM()

        # # 通道放缩
        self.mlp = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels * mlp_ratio, 1),
            nonlinearity,
            nn.Conv2d(in_channels * mlp_ratio, out_channels, 1)
        )
        
        # HW通道注意力
        self.dar = DAR(in_channels)
        
        # # 通道注意力1x1卷积
        self.channels_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(in_channels),
            nonlinearity)
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(in_channels),
            nonlinearity)
        
        self.concat_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(in_channels),
            nonlinearity)

    def forward(self, x1, x2):
        
        # x1: 时相1的输入
        # x2: 时相2的输入
        assert x1.shape == x2.shape, "Input tensors must have the same shape."
        B, C, H, W = x1.shape
        
        
        #########################################################################
        ## 得到通道注意力
        #########################################################################
        x_channels = self.dar(x1, x2)


        #########################################################################
        ## 得到空间注意力
        #########################################################################

        # 得到融合时相信息
        x_concat = torch.abs(x2 - x1)
        # 融合时相信息的1x1Conv+SiLU
        x_concat = self.concat_conv(x_concat)

        # 时相1 W方向窗口化 + DWConv+ SS2D + LN
        x1_w_window = self.dw_w_1_ss2d(im2cswin(x1, h_sp=1, w_sp=self.w_sp))
        # 时相2 W方向窗口化 + DWConv+ SS2D + LN
        x2_w_window = self.dw_w_2_ss2d(im2cswin(x2, h_sp=1, w_sp=self.w_sp))

        # 时相1 H方向窗口化 + DWConv + SS2D + LN
        x1_h_window = self.dw_h_1_ss2d(im2cswin(x1, h_sp=self.h_sp, w_sp=1))
        # 时相2 H方向窗口化 + DWConv + SS2D + LN
        x2_h_window = self.dw_h_2_ss2d(im2cswin(x2, h_sp=self.h_sp, w_sp=1))

        # 融合时相信息窗口化
        xc_w_window = im2cswin(x_concat, h_sp=1, w_sp=self.w_sp)
        xc_h_window = im2cswin(x_concat, h_sp=self.h_sp, w_sp=1)

        # 时相1 H窗口化特征 与 融合时相H窗口化特征 相乘
        x1_h_window = xc_h_window * x1_h_window
        # 时相2 H窗口化特征 与 融合时相H窗口化特征 相乘
        x2_h_window = xc_h_window * x2_h_window

        # 时相1 W窗口化特征 与 融合时相W窗口化特征 相乘
        x1_w_window = xc_w_window * x1_w_window
        # 时相2 W窗口化特征 与 融合时相W窗口化特征 相乘
        x2_w_window = xc_w_window * x2_w_window

        # 时相1 H窗口化特征复原
        x1_h = cswin2im(x1_h_window, h_sp=self.h_sp, w_sp=1, b=B)
        # 时相1 W窗口化特征复原
        x1_w = cswin2im(x1_w_window, h_sp=1, w_sp=self.w_sp, b=B)

        # 时相2 H窗口化特征复原
        x2_h = cswin2im(x2_h_window, h_sp=self.h_sp, w_sp=1, b=B)
        # 时相2 W窗口化特征复原
        x2_w = cswin2im(x2_w_window, h_sp=1, w_sp=self.w_sp, b=B)

        # 得到空间注意力结果
        x_h = x2_h + x1_h
        x_w = x2_w + x1_w
        x_spatial = x_h + x_w

        # # 不同时相信息整合
        # 1. 1x1conv + 1x1conv
        x_ = self.mlp(self.channels_conv(x_channels) + self.spatial_conv(x_spatial))
        
        #2. add
        x_ = x_channels + x_spatial
        
        # 3. concat + 1x1conv
        x_ = self.concat_fusion(torch.cat([x_channels, x_spatial], dim=1))
        
        x_ = self.mlp(x_spatial)
        x_ = self.channels_conv(x_channels)
        
        return x_channels


class SP4CD_S(nn.Module):
    # 串行HW操作
    def __init__(self, in_channels, out_channels, h_sp, w_sp, mlp_ratio=4, mode_s='s1s2', nonlinearity=nn.SiLU()):
        
        super(SP4CD_S, self).__init__()
        self.h_sp = h_sp
        self.w_sp = w_sp
        self.mode_s = mode_s

        # 融合时相信息Mamba_Conv
        self.concat_conv = SimAM()
        
        # HW通道注意力
        self.dar = DAR(in_channels)
        
        # 通道注意力1x1卷积
        self.channels_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(in_channels),
            nonlinearity)
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(in_channels),
            nonlinearity)

        # 时相1 H方向DW卷积+SS2D
        self.dw_h_1_ss2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=autopad(3), groups=in_channels),
            SS2D(d_model=in_channels // 2),
            LayerNorm2d(in_channels))

        # 时相1 W方向DW卷积+SS2D
        self.dw_w_1_ss2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=autopad(3), groups=in_channels),
            SS2D(d_model=in_channels // 2),
            LayerNorm2d(in_channels))

        # 时相1 H窗口 与 融合特征相乘后 的 SimAM
        self.concat_conv_2_h = SimAM()

        # 时相1 W窗口 与 融合特征相乘后 的 SimAM
        self.concat_conv_2_w = SimAM()

        # 时相2 H方向DW卷积+SS2D
        self.dw_h_2_ss2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=autopad(3), groups=in_channels),
            SS2D(d_model=in_channels // 2),
            LayerNorm2d(in_channels))

        # 时相2 W方向DW卷积+SS2D
        self.dw_w_2_ss2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=autopad(3), groups=in_channels),
            SS2D(d_model=in_channels // 2),
            LayerNorm2d(in_channels))

        # 通道放缩
        self.mlp = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels * mlp_ratio, 1),
            nonlinearity,
            nn.Conv2d(in_channels * mlp_ratio, out_channels, 1)
        )
    def forward_s1s2(self, x1, x2):
        # x1: 时相1的输入
        # x2: 时相2的输入
        assert x1.shape == x2.shape, "Input tensors must have the same shape."
        B, C, H, W = x1.shape
        
        x_channels = self.dar(x1, x2)

        # 融合时相信息
        x_concat = torch.abs(x2 - x1)
        # 融合时相信息的1x1Conv+SiLU
        x_concat = self.concat_conv(x_concat)

        # 时相1 W方向窗口化 + DWConv+ SS2D + LN
        x1_w_window = self.dw_w_1_ss2d(im2cswin(x1, h_sp=1, w_sp=self.w_sp))
        # 时相1 H方向窗口化 + DWConv + SS2D + LN
        x1_h_window = self.dw_h_1_ss2d(im2cswin(x1, h_sp=self.h_sp, w_sp=1))

        # 时相2 W方向窗口化 + DWConv+ SS2D + LN
        x2_w_window = self.dw_w_2_ss2d(im2cswin(x2, h_sp=1, w_sp=self.w_sp))
        # 时相2 H方向窗口化 + DWConv + SS2D + LN
        x2_h_window = self.dw_h_2_ss2d(im2cswin(x2, h_sp=self.h_sp, w_sp=1))

        # 融合时相信息窗口化
        xc_w_window = im2cswin(x_concat, h_sp=1, w_sp=self.w_sp)
        xc_h_window = im2cswin(x_concat, h_sp=self.h_sp, w_sp=1)

        # 融合时相信息与时相1 H窗口化特征相乘
        x1_h_window = xc_h_window * x1_h_window
        # 融合时相信息与时相1 W窗口化特征相乘
        x1_w_window = xc_w_window * x1_w_window

        # 相乘后的特征通过第二阶段的mamba 1x1Conv + SiLU
        x1_h_window = self.concat_conv_2_h(x1_h_window)
        x1_w_window = self.concat_conv_2_w(x1_w_window)

        # 第二阶段的特征相乘
        x2_h_window = x1_h_window * x2_h_window
        x2_w_window = x1_w_window * x2_w_window

        # 第二阶段的特征复原
        x_w = cswin2im(x2_w_window, h_sp=1, w_sp=self.w_sp, b=B)
        x_h = cswin2im(x2_h_window, h_sp=self.h_sp, w_sp=1, b=B)

        x_spatial = x_w + x_h
        
        x_ = self.mlp(self.channels_conv(x_channels) + self.spatial_conv(x_spatial))
        
        x_ = x_channels

        return x_
    
    def forward_s2s1(self, x1, x2):
        # x1: 时相1的输入
        # x2: 时相2的输入
        assert x1.shape == x2.shape, "Input tensors must have the same shape."
        B, C, H, W = x1.shape
        
        x_channels = self.dar(x1, x2)

        # 融合时相信息
        x_concat = torch.abs(x2 - x1)
        
        # 融合时相信息的1x1Conv+SiLU
        x_concat = self.concat_conv(x_concat)

        # 时相1 W方向窗口化 + DWConv+ SS2D + LN
        x1_w_window = self.dw_w_1_ss2d(im2cswin(x1, h_sp=1, w_sp=self.w_sp))
        # 时相1 H方向窗口化 + DWConv + SS2D + LN
        x1_h_window = self.dw_h_1_ss2d(im2cswin(x1, h_sp=self.h_sp, w_sp=1))

        # 时相2 W方向窗口化 + DWConv+ SS2D + LN
        x2_w_window = self.dw_w_2_ss2d(im2cswin(x2, h_sp=1, w_sp=self.w_sp))
        # 时相2 H方向窗口化 + DWConv + SS2D + LN
        x2_h_window = self.dw_h_2_ss2d(im2cswin(x2, h_sp=self.h_sp, w_sp=1))

        # 融合时相信息窗口化
        xc_w_window = im2cswin(x_concat, h_sp=1, w_sp=self.w_sp)
        xc_h_window = im2cswin(x_concat, h_sp=self.h_sp, w_sp=1)

        # 融合时相信息与时相2 H窗口化特征相乘
        x2_h_window = xc_h_window * x2_h_window
        # 融合时相信息与时相2 W窗口化特征相乘
        x2_w_window = xc_w_window * x2_w_window

        # 相乘后的特征通过第二阶段的mamba 1x1Conv + SiLU
        x2_h_window = self.concat_conv_2_h(x2_h_window)
        x2_w_window = self.concat_conv_2_w(x2_w_window)

        # 第二阶段的特征相乘
        # 融合时相信息与时相1 H窗口化特征相乘
        x2_h_window = x1_h_window * x2_h_window
        # 融合时相信息与时相1 W窗口化特征相乘
        x2_w_window = x1_w_window * x2_w_window

        # 第二阶段的特征复原
        x_w = cswin2im(x2_w_window, h_sp=1, w_sp=self.w_sp, b=B)
        x_h = cswin2im(x2_h_window, h_sp=self.h_sp, w_sp=1, b=B)

        x_spatial = x_w + x_h
        
        x_ = self.mlp(self.channels_conv(x_channels) + self.spatial_conv(x_spatial))

        return x_
        
    def forward(self, x1, x2):
        if self.mode_s == 's2s1':
            x_ = self.forward_s2s1(x1, x2)
        elif self.mode_s == 's1s2':
            x_ = self.forward_s1s2(x1, x2)
        else:
            raise ValueError("mode_s must be either 's2s1' or 's1s2'.")
        return x_


if __name__ == '__main__':
    model = SP4CD_P(in_channels=64, out_channels=64, h_sp=4, w_sp=4, mlp_ratio=4).cuda()
    x1 = torch.randn(1, 64, 56, 56).cuda()
    x2 = torch.randn(1, 64, 56, 56).cuda()
    
    from thop import profile
    flops, params = profile(model, inputs=(x1, x2), verbose=False)
    print(f'FLOPs: {flops / 1e9:.2f} G')
    print(f'Params: {params / 1e6:.2f} M')