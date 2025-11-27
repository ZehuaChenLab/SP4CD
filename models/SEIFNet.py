import torch
import torch.nn as nn
from .seifnet_help.resnet import resnet18

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_h = a_h.expand(-1,-1,h,w)
        a_w = a_w.expand(-1, -1, h, w)

        # out = identity * a_w * a_h

        return a_w , a_h


class CoDEM2(nn.Module):
    '''
    最新的版本
    '''
    def __init__(self,channel_dim):
        super(CoDEM2, self).__init__()

        self.channel_dim=channel_dim

        #特征连接后
        self.Conv3 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=2*self.channel_dim,kernel_size=3,stride=1,padding=1)
        #特征加和后
        # self.AvgPool = nn.functional.adaptive_avg_pool2d()
        self.Conv1 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)
        #最后输出
        # self.Conv1_ =nn.Conv2d(in_channels=3*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)
        self.BN1 = nn.BatchNorm2d(2*self.channel_dim)
        self.BN2 = nn.BatchNorm2d(self.channel_dim)
        self.ReLU = nn.ReLU(inplace=True)
        #我的注意力机制
        self.coAtt_1 = CoordAtt(inp=channel_dim, oup=channel_dim, reduction=16)
        #通道,kongjian注意力机制
        # self.cam =ChannelAttention(in_channels=self.channel_dim,ratio=16)
        # self.sam = SpatialAttention()

    def forward(self,x1,x2):
        B,C,H,W = x1.shape
        f_d = torch.abs(x1-x2) #B,C,H,W
        f_c = torch.cat((x1, x2), dim=1)  # B,2C,H,W
        z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))

        d_aw, d_ah = self.coAtt_1(f_d)
        z_d = f_d * d_aw * d_ah


        out = z_d + z_c

        return out

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )

class Backbone(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5,
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Backbone, self).__init__()


        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        #
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        #
        # self.resnet_stages_num = resnet_stages_num
        #
        self.if_upsample_2x = if_upsample_2x

        #
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()



#Backbone的区别：提取特征差异
    # BIT用的
    def forward_single0(self, x):

        x = self.backbone(x)

        x0, x1, x2, x3 = x
        if self.if_upsample_2x:
            x = self.upsamplex2(x2)
        else:
            x = x2
        # output layers
        x = self.conv_pred(x)
        return x

    # SEIFNet用
    def forward_single(self, x):

        f= self.backbone(x)

        return f

    def forward_down(self,x):
        f = self.downsample(x)
        return f

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_pool_out = self.avg_pool(x)
        max_out_out = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_out_out)))
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class ACFF2(nn.Module):
    '''
    最新版本的ACFF 4.21,将cat改成+，去掉卷积
    '''
    def __init__(self, channel_L, channel_H):
        super(ACFF2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel_H,out_channels=channel_L,kernel_size=1, stride=1,padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Conv2d(in_channels=2*channel_L, out_channels=channel_L, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(channel_L)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(in_channels=channel_L,ratio=16)

    def forward(self, f_low,f_high):
        # _,c,h,w = f_low.shape
        #f4上采样，通道数变成原来的1/2,长宽变为原来的2倍
        f_high = self.relu(self.BN(self.conv1(self.up(f_high))))

        f_cat = f_high + f_low

        adaptive_w = self.ca(f_cat)

        out = f_low * adaptive_w+f_high*(1-adaptive_w) # B,C_l,h,w
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d

        self.cbam = CBAM(channel = self.mid_d)

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        context = self.cbam(x)

        x_out = self.conv2(context)

        return x_out

class SEIFNet(Backbone):
    """
    4.4 最新版本，改进了Diff（DEM2_sobel)和ACFF2

    """

    def __init__(self, input_nc, output_nc,
                 decoder_softmax=False, embed_dim=64,
                 Building_Bool=False):
        super(SEIFNet, self).__init__(input_nc, output_nc)

        self.stage_dims = [64, 128, 256, 512]
        self.output_nc=output_nc
        self.backbone = resnet18(pretrained=True)


        self.diff1 = CoDEM2(self.stage_dims[0])
        self.diff2  = CoDEM2(self.stage_dims[1])
        self.diff3  = CoDEM2(self.stage_dims[2])
        self.diff4  = CoDEM2(self.stage_dims[3])



        #decoder
        self.ACFF3 = ACFF2(channel_L=self.stage_dims[2], channel_H=self.stage_dims[3])
        self.ACFF2 = ACFF2(channel_L=self.stage_dims[1], channel_H=self.stage_dims[2])
        self.ACFF1 = ACFF2(channel_L=self.stage_dims[0], channel_H=self.stage_dims[1])
        #(修改了一下cbam)
        self.sam_p4 = SupervisedAttentionModule(self.stage_dims[3])
        self.sam_p3 = SupervisedAttentionModule(self.stage_dims[2])
        self.sam_p2 = SupervisedAttentionModule(self.stage_dims[1])
        self.sam_p1 = SupervisedAttentionModule(self.stage_dims[0])



        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8,mode='bilinear')
        # self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear')


        self.conv4 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)

        self.conv_final1 = nn.Conv2d(64, output_nc, kernel_size=1)


    def forward(self, x1, x2):


        #res18
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        x1_0,x1_1,x1_2,x1_3 = f1
        x2_0,x2_1,x2_2,x2_3 = f2

        #diff_last
        d1 = self.diff1(x1_0, x2_0)
        d2 = self.diff2(x1_1, x2_1)
        d3 = self.diff3(x1_2, x2_2)
        d4 = self.diff4(x1_3, x2_3)

        p4 = self.sam_p4(d4)

        ACFF_43 = self.ACFF3(d3,p4)
        p3 = self.sam_p3(ACFF_43)

        ACFF_32 =self.ACFF2(d2,p3)
        p2 = self.sam_p2(ACFF_32)

        ACFF_21 = self.ACFF1(d1,p2)
        p1 = self.sam_p1(ACFF_21)

        p4_up = self.upsample8(p4)
        p4_up =self.conv4(p4_up)

        p3_up = self.upsample4(p3)
        p3_up = self.conv3(p3_up)

        p2_up = self.upsample2(p2)
        p2_up = self.conv2(p2_up)

        p= p1+p2_up+p3_up+p4_up

        p_up =self.upsample4(p)

        output = self.conv_final1(p_up)



        return output

if __name__ == "__main__":
    mscanet = SEIFNet(input_nc=3, output_nc=2)
    # img = torch.rand([8, 3, 512, 512])
    #
    # res = mscanet(img, img)
    # print(res[0].shape)
    # print(res[1].shape)
    # print(res[2].shape)
    from thop import profile

    # caculate flops1
    input1 = torch.randn(1, 3, 256, 256)
    flops1, params1 = profile(mscanet, inputs=(input1, input1))
    print("flops=G", flops1 / (1000 ** 3))
    print("parms=M", params1 / (1000 ** 2))