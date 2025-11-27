
import sys
sys.path.append('.')
sys.path.append('..')
from .SS2D import SS2D
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple
from .SimAM import SimAM
from .DAR import DAR

nonlinearity = nn.SiLU()

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


class HW_Block_Parallel(nn.Module):
    # 并行HW操作
    def __init__(self, in_channels, out_channels, h_sp, w_sp, mlp_ratio=4, nonlinearity=nn.SiLU()):
        
        super(HW_Block_Parallel, self).__init__()
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
        
        # 通道注意力1x1卷积
        self.channels_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(in_channels),
            nonlinearity)
        
        # self.spatial_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        #     LayerNorm2d(in_channels),
        #     nonlinearity)
        
        # self.concat_fusion = nn.Sequential(
        #     nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0),
        #     LayerNorm2d(in_channels),
        #     nonlinearity)

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

        # 不同时相信息整合
        # 1. 1x1conv + 1x1conv
        x_ = self.mlp(self.channels_conv(x_channels) + self.spatial_conv(x_spatial))
        
        # 2. add
        # x_ = x_channels + x_spatial
        
        # # 3. concat + 1x1conv
        # x_ = self.concat_fusion(torch.cat([x_channels, x_spatial], dim=1))
        
        # x_ = self.mlp(x_spatial)
        # x_ = self.channels_conv(x_channels)
        
        return x_


class HW_Block_Series(nn.Module):
    # 串行HW操作
    def __init__(self, in_channels, out_channels, h_sp, w_sp, mlp_ratio=4, mode_s='s1s2', nonlinearity=nn.SiLU()):
        
        super(HW_Block_Series, self).__init__()
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
    model = HW_Block_Parallel(in_channels=64, out_channels=64, h_sp=4, w_sp=4, mlp_ratio=4).cuda()
    x1 = torch.randn(1, 64, 56, 56).cuda()
    x2 = torch.randn(1, 64, 56, 56).cuda()
    
    from thop import profile
    flops, params = profile(model, inputs=(x1, x2), verbose=False)
    print(f'FLOPs: {flops / 1e9:.2f} G')
    print(f'Params: {params / 1e6:.2f} M')