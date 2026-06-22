#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F #SAFM所需
from torch.nn.parameter import Parameter #CNN所需
from einops import rearrange  # LAWDS所需
import math




class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # use same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        '''
        x --> Conv2d --> BN --> activation --> x
        '''
        # Ensure x is 4D before passing to Conv2d
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension, now x shape will be [1, channels, height, width]
        return self.act(self.bn(self.conv(x)))      # Conv ==> BN ==> activate

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv (with BN and activation) + Pointwise Conv (with BN and activation)"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,     # depthwise
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        '''
        x --> dconv (e.g. depthwise conv --> BN --> act) --> pconv (e.g. pointwise conv --> BN --> act) --> x
        '''
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        '''
          | --> BaseConv (in_channels to hidden_channels) --> Conv (hidden_channels to out_channels) \
        x                                                                                              --> add(optional)
          \ --> shortcut                                                                             |
        '''
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        '''
          | --> BaseConv(in_channels, mid_channels) --> BaseConv(mid_channels, in_channels) \
        x                                                                                     --> add --> x
          \ --> shortcut                                                                    |
        '''
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        '''
                       | shortcut  \
                       | MaxPool2d \
        x --> BaseConv              --> BaseConv --> x
                       \ MaxPool2d |
                       \ MaxPool2d |
        '''
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        '''
             | BaseConv --> Bottleneck * n \
        x -->                               cat --> BaseConv
             \ BaseConv                    |
        '''
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class Upsample(nn.Module):
    """Unsample layer with pixel_shuffle with/without Convs"""

    def __init__(self, in_channels, out_channels, gain=2, conv=True):
        super().__init__()
        self.gain = gain
        if not conv:
            assert in_channels * (gain ** 2) == out_channels
            self.proj = nn.Identity()
        else:
            self.proj = BaseConv(in_channels // 4, out_channels, 3, 1, act="silu")
        
    def forward(self, x):
        x = nn.functional.pixel_shuffle(x, self.gain)
        return self.proj(x)
    
#---------------------------------------------------------------------
# class AutomaticWeightedLoss(nn.Module):  #计算自适应loss，在yolo-head中修改
#     """automatically weighted multi-task loss
#     Params：
#         num: int，the number of loss
#         x: multi-task loss
#     Examples：
#         loss1=1
#         loss2=2
#         awl = AutomaticWeightedLoss(2)
#         loss_sum = awl(loss1, loss2)
#     """
#     def __init__(self, num=2):
#         super(AutomaticWeightedLoss, self).__init__()
#         params = torch.ones(num, requires_grad=True)
#         self.params = torch.nn.Parameter(params)
 
#     def forward(self, *x):
#         loss_sum = 0
#         for i, loss in enumerate(x):
#             loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
#         return loss_sum

#         # id_loss = loss_id * self.s_moco + loss_id_aux * (1. - self.s_moco)#计算最终的 ID 损失，self.s_moco 是一个权重参数，用于平衡这两种损失。
#         # #loss = self.s_bbox * det_loss + self.s_reid * id_loss#计算最终的损失，包括检测损失和 ID 损失。
#         # awl = AutomaticWeightedLoss(2)   # we have 2 losses       
#         # loss = awl(det_loss, id_loss)
class AutomaticWeightedLoss(nn.Module):        #改进版，AutomaticWeightedLoss类初始化时会创建一个参数列表，这些参数用于自适应调节各任务损失的权重。l2_reg参数用于控制L2正则化的强度，可以帮助避免参数过度放大。每个损失的权重通过 weight = torch.exp(-self.params[i]) 计算得出，这确保了权重总是正数且通过指数函数动态调整。L2正则化项reg_loss加到总损失中以防止权重参数的无限增长。
    """
    Automatically weighted multi-task loss with L2 regularization and dynamic adjustment.
    Params:
        num: int, the number of losses
        l2_reg: float, L2 regularization strength
        initial_value: float, initial value for the parameters
    """
    def __init__(self, num=2, l2_reg=0.01, initial_value=1.0):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.full((num,), initial_value, requires_grad=True)
        self.params = nn.Parameter(params)
        self.l2_reg = l2_reg

    def forward(self, *losses):
        loss_sum = 0
        reg_loss = 0
        for i, loss in enumerate(losses):
            weight = torch.exp(-self.params[i])
            loss_sum += weight * loss
            reg_loss += self.l2_reg * self.params[i] ** 2  # L2 regularization

        total_loss = loss_sum + reg_loss
        return total_loss
# class ECAAttention(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """

#     def __init__(self, c1, k_size=3):
#         super(ECAAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         # Multi-scale information fusion
#         y = self.sigmoid(y)

#         return x * y.expand_as(x)       #用于将全局注意力图 y 扩展为与输入特征图 x 相同的形状，以便进行逐元素乘法操作。

class ECAAttention(nn.Module):  #自适应卷积核ECA
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        gamma: Coefficient of kernel size computation
        b: Offset of kernel size computation
    """
    def __init__(self, c1, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 计算自适应卷积核大小
        t = int(abs(math.log2(c1)/gamma + b))
        k_size = t if t % 2 else t + 1  # 确保核大小为奇数
        
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out

        #return self.sigmoid(out)

        out = self.sigmoid(out)
        out = out * x
        out = out + x
        
        return out
    
class AdaptiveAvgPoolMLP(nn.Module):       #自建模块，自适应平均池化和 MLP 模块
    def __init__(self, input_channels, output_size, hidden_size):
        """
        初始化自适应平均池化和 MLP 模块。
        
        参数:
        - input_channels: 输入特征图的通道数
        - output_size: MLP 的输出维度
        - hidden_size: MLP 中隐藏层的尺寸
        """
        super(AdaptiveAvgPoolMLP, self).__init__()
        
        # 自适应平均池化层
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 定义 MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_size),
            nn.GELU(),
            #nn.Linear(hidden_size, output_size)
            nn.Linear(hidden_size, input_channels)
        )
    
    def forward(self, x):
        """
        前向传播函数。
        
        参数:
        - x: 输入张量，形状为 (batch_size, input_channels, height, width)
        
        返回:
        - 输出张量，形状为 (batch_size, output_size)
        """
        # 应用自适应平均池化
        y = self.adaptive_avg_pool(x)
        # 将输出展平
        y = y.view(y.size(0), -1)
        # 通过 MLP
        y = self.mlp(y)
        y = y.unsqueeze(-1).unsqueeze(-1).expand_as(x) # 将 MLP 的输出扩展到与输入特征图相同的高度和宽度
        out = x * y
        out = out + x

        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        out = x * y
        out = out + x
        return out


class LAWDS(nn.Module):
    # Light Adaptive-weight downsampling
    def __init__(self, ch, group=16) -> None:
        super().__init__()
        
        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            #Conv(ch, ch, k=1)
            BaseConv(ch, ch, ksize=1, stride=1) # 将Conv改为BaseConv
        )
        
        #self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))
        self.ds_conv = BaseConv(ch, ch * 4, ksize=3, stride=2, groups=(ch // group)) # 将Conv改为BaseConv
        
    
    def forward(self, x):
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)
        
        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        return x

# class CCN(nn.Module):                       #改进CCN模块           合并重识别和检测特征
#     def __init__(self,k_size = 3,ch=()):
#         super(CCN, self).__init__()
#         #self.independence = 0.7
#         #self.share = 0.3
#         #self.w1 = Parameter(torch.ones(1)*0.5)
#         #self.w2 = Parameter(torch.ones(1)*0.5)
#                                                                     # Adaptive weighting layers
#         self.adaptive_weighting = nn.Sequential(
#             nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(ch, 2, kernel_size=1, stride=1, padding=0),  # Learn two weights for both tasks
#             nn.Sigmoid()
#         )
#         w = 6
#         h = 10
#         self.avg_pool = nn.AdaptiveAvgPool2d((w,h))

#         self.c_attention1 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                                           nn.InstanceNorm2d(num_features=ch),
#                                           nn.LeakyReLU(0.3, inplace=True))
#         self.c_attention2 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                                           nn.InstanceNorm2d(num_features=ch),
#                                           nn.LeakyReLU(0.3, inplace=True))


#         self.sigmoid = nn.Sigmoid()
#         #self.conv1 = Conv(ch, ch, k=1)
#         #self.conv2 = Conv(ch, ch, k=1)

#     def forward(self, x):
#         # x: input features with shape [b, c, h, w]
#         b, c, h, w = x.size()

#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)

#         y_t1 = self.c_attention1(y)
#         y_t2 = self.c_attention2(y)
#         bs,c,h,w = y_t1.shape
#         y_t1 =y_t1.view(bs, c, h*w)
#         y_t2 =y_t2.view(bs, c, h*w)

#         y_t1_T = y_t1.permute(0, 2, 1)
#         y_t2_T = y_t2.permute(0, 2, 1)
#         M_t1 = torch.matmul(y_t1, y_t1_T)
#         M_t2 = torch.matmul(y_t2, y_t2_T)
#         M_t1 = F.softmax(M_t1, dim=-1)
#         M_t2 = F.softmax(M_t2, dim=-1)

#         M_s1 = torch.matmul(y_t1, y_t2_T)
#         M_s2 = torch.matmul(y_t2, y_t1_T)
#         M_s1 = F.softmax(M_s1, dim=-1)
#         M_s2 = F.softmax(M_s2, dim=-1)

#         x_t1 = x
#         #x_t2 = x
#         bs,c,h,w = x_t1.shape
#         x_t1 = x_t1.contiguous().view(bs, c, h*w)
#         #x_t2 = x_t2.contiguous().view(bs, c, h*w)

#         # x_t1 = torch.matmul(self.w1*M_t1 + (1-self.w1)*M_s1, x_t1).contiguous().view(bs, c, h, w)
#         # x_t2 = torch.matmul(self.w2*M_t2 + (1-self.w2)*M_s2, x_t2).contiguous().view(bs, c, h, w)

#         weights = self.adaptive_weighting(x)  # Learnable adaptive weights
#         w1, w2 = weights.split(1, dim=1)  # Split learned weights for each task
#         w1 = w1.view(-1, 1, 1, 1)  # Reshape for broadcasting
#         w2 = w2.view(-1, 1, 1, 1)

#          # Combine both task-specific matrices
#         #combined_M = self.w1 * M_t1 + (1 - self.w1) * M_s1 + self.w2 * M_t2 + (1 - self.w2) * M_s2
#         combined_M = w1 * M_t1 + (1 - w1) * M_s1 + w2 * M_t2 + (1 - w2) * M_s2


#         # Apply the combined attention matrix to the feature map
#         x_combined = torch.matmul(combined_M, x_t1).contiguous().view(bs, c, h, w)
#         # Apply the combined attention matrix to the feature map
#         #x_combined = torch.matmul(combined_M, x_t1).permute(0, 2, 1).contiguous().view(bs, c, h, w)

#         return x_combined + x



# class CCN(nn.Module):                       #CSTrack中的REN模块，为重识别和检测分配不同特征    报错
#     def __init__(self,k_size = 3,ch=()):
#         super(CCN, self).__init__()
#         #self.independence = 0.7
#         #self.share = 0.3
#         self.w1 = Parameter(torch.ones(1)*0.5)
#         self.w2 = Parameter(torch.ones(1)*0.5)
#         # Adaptive weighting layers
#         # self.adaptive_weighting = nn.Sequential(
#         #     nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
#         #     nn.ReLU(),
#         #     nn.Conv2d(ch, 2, kernel_size=1, stride=1, padding=0),  # Learn two weights for both tasks
#         #     nn.Sigmoid()
#         # )
#         w = 6
#         h = 10
#         self.avg_pool = nn.AdaptiveAvgPool2d((w,h))

#         self.c_attention1 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                                           nn.InstanceNorm2d(num_features=ch),
#                                           nn.LeakyReLU(0.3, inplace=True))
#         self.c_attention2 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                                           nn.InstanceNorm2d(num_features=ch),
#                                           nn.LeakyReLU(0.3, inplace=True))


#         self.sigmoid = nn.Sigmoid()
#         #self.conv1 = Conv(ch, ch, k=1)
#         #self.conv2 = Conv(ch, ch, k=1)

#     def forward(self, x):
#         # x: input features with shape [b, c, h, w]
#         b, c, h, w = x.size()

#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)

#         y_t1 = self.c_attention1(y)
#         y_t2 = self.c_attention2(y)
#         bs,c,h,w = y_t1.shape
#         y_t1 =y_t1.view(bs, c, h*w)
#         y_t2 =y_t2.view(bs, c, h*w)

#         y_t1_T = y_t1.permute(0, 2, 1)
#         y_t2_T = y_t2.permute(0, 2, 1)
#         M_t1 = torch.matmul(y_t1, y_t1_T)
#         M_t2 = torch.matmul(y_t2, y_t2_T)
#         M_t1 = F.softmax(M_t1, dim=-1)
#         M_t2 = F.softmax(M_t2, dim=-1)

#         M_s1 = torch.matmul(y_t1, y_t2_T)
#         M_s2 = torch.matmul(y_t2, y_t1_T)
#         M_s1 = F.softmax(M_s1, dim=-1)
#         M_s2 = F.softmax(M_s2, dim=-1)

#         x_t1 = x
#         x_t2 = x
#         bs,c,h,w = x_t1.shape
#         x_t1 = x_t1.contiguous().view(bs, c, h*w)
#         x_t2 = x_t2.contiguous().view(bs, c, h*w)

#         #x_t1 = torch.matmul(self.independence*M_t1 + self.share*M_s1, x_t1).contiguous().view(bs, c, h, w)
#         #x_t2 = torch.matmul(self.independence*M_t2 + self.share*M_s2, x_t2).contiguous().view(bs, c, h, w)
#         x_t1 = torch.matmul(self.w1*M_t1 + (1-self.w1)*M_s1, x_t1).contiguous().view(bs, c, h, w)
#         x_t2 = torch.matmul(self.w2*M_t2 + (1-self.w2)*M_s2, x_t2).contiguous().view(bs, c, h, w)

#         out = x_t1 + x_t2
#         out = out.contiguous()  # Ensure it's contiguous if needed
#         out = out.view(bs, c, h, w) 

#         #out = x_t1+x_t2
#         #print("M_t1",torch.sort(M_t1[0][0]))
#         #print("y_t1",torch.max(y_t1),torch.min(y_t1))
#         #print("y_t2", torch.max(y_t2), torch.min(y_t2))
#         #return [x_t1+x,x_t2+x]
#         return out

# class CCN(nn.Module):  # CSTrack中的REN模块，为重识别和检测分配不同特征       简单加权的方法
#     def __init__(self, k_size=3, ch=()):
#         super(CCN, self).__init__()
#         self.w1 = Parameter(torch.ones(1) * 0.5)
#         self.w2 = Parameter(torch.ones(1) * 0.5)
#         w = 6
#         h = 10
#         self.avg_pool = nn.AdaptiveAvgPool2d((w, h))

#         self.c_attention1 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                                           nn.InstanceNorm2d(num_features=ch),
#                                           nn.LeakyReLU(0.3, inplace=True))
#         self.c_attention2 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                                           nn.InstanceNorm2d(num_features=ch),
#                                           nn.LeakyReLU(0.3, inplace=True))

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # x: input features with shape [b, c, h, w]
#         b, c, h, w = x.size()

#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)

#         y_t1 = self.c_attention1(y)
#         y_t2 = self.c_attention2(y)
#         bs, c, h, w = y_t1.shape
#         y_t1 = y_t1.view(bs, c, h * w)
#         y_t2 = y_t2.view(bs, c, h * w)

#         y_t1_T = y_t1.permute(0, 2, 1)
#         y_t2_T = y_t2.permute(0, 2, 1)
#         M_t1 = torch.matmul(y_t1, y_t1_T)
#         M_t2 = torch.matmul(y_t2, y_t2_T)
#         M_t1 = F.softmax(M_t1, dim=-1)
#         M_t2 = F.softmax(M_t2, dim=-1)

#         M_s1 = torch.matmul(y_t1, y_t2_T)
#         M_s2 = torch.matmul(y_t2, y_t1_T)
#         M_s1 = F.softmax(M_s1, dim=-1)
#         M_s2 = F.softmax(M_s2, dim=-1)

#         x_t1 = x
#         x_t2 = x
#         bs, c, h, w = x_t1.shape
#         x_t1 = x_t1.contiguous().view(bs, c, h * w)
#         x_t2 = x_t2.contiguous().view(bs, c, h * w)


#         # Combine both task-specific matrices
#         combined_M = self.w1 * M_t1 + (1 - self.w1) * M_s1 + self.w2 * M_t2 + (1 - self.w2) * M_s2
        

#         # Apply the combined attention matrix to the feature map
#         x_combined = torch.matmul(combined_M, x_t1).contiguous().view(bs, c, h, w)

#         return x_combined + x
class NECKChannelAttention(nn.Module):
    def __init__(self, c1, ratio=16):
        super(NECKChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(c1, c1 // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(c1 // 16, c1, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out

        #return self.sigmoid(out)

        out = self.sigmoid(out)
        out = out * x
        out = out + x
        
        return out

class NECKSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(NECKSpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        out = x * y
        out = out + x
        return out
class AdvancedChannelAttention(nn.Module):
    def __init__(self, c1, ratio=16):
        super(AdvancedChannelAttention, self).__init__()
        
        # 双路特征提取
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 主干特征变换
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(c1, c1 // ratio, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(c1 // ratio, c1, 1, bias=False)
        )
        
        # 通道交互模块
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(c1, c1, 1, groups=4, bias=False),  # 分组卷积增强通道交互
            nn.SiLU(),
            nn.Conv2d(c1, c1, 1, bias=False)
        )
        
        # 自适应权重
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        identity = x
        
        # 1. 主干特征提取
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        main_attn = torch.sigmoid(avg_out + max_out)
        
        # 2. 通道交互
        mixed = self.channel_mixer(x * main_attn)
        mixed_attn = torch.sigmoid(mixed)
        
        # 3. 自适应特征增强
        out = x * main_attn * (1 + self.beta) + x * mixed_attn * self.gamma
        
        return out + identity

