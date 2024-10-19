# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

from __future__ import print_function, division, absolute_import
from math import ceil
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
# from ops.hs_resnet import HSBlock
from ops.base_model_v2 import TokenLearnerModule, TokenLearnerModule1D, FSAttention, TemporalAttention



__all__ = ['FBResNet', 'fbres2net50','fbresnet50', 'fbresnet101']

model_urls = {
        'fbresnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet50-19c8e357.pth',
        'fbresnet101': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet101-5d3b4d8f.pth',
        'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
        'res2net50_48w_2s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_48w_2s-afed724a.pth',
        'res2net50_14w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_14w_8s-6527dddc.pth',
        'res2net50_26w_6s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_6s-19041792.pth',
        'res2net50_26w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_8s-2c7c9f12.pth',
        'res2net101_26w_4s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net101_26w_4s-02a759a1.pth',
}

class mSEModule(nn.Module):
    def __init__(self, channel, n_segment=8,index=1):
        super(mSEModule, self).__init__()
        self.channel = channel
        self.reduction = 16
        self.n_segment = n_segment
        self.stride = 2**(index-1)
        self.conv1 = nn.Conv2d(in_channels=self.channel,
                out_channels=self.channel//self.reduction,
                kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        self.conv2 = nn.Conv2d(in_channels=self.channel//self.reduction,
                out_channels=self.channel//self.reduction,
                kernel_size=3, padding=1, groups=self.channel//self.reduction, bias=False)

        self.avg_pool_forward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_forward4 = nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.sigmoid_forward = nn.Sigmoid()

        self.avg_pool_backward2 = nn.AvgPool2d(kernel_size=2, stride=2)#nn.AdaptiveMaxPool2d(1)
        self.avg_pool_backward4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.sigmoid_backward = nn.Sigmoid()

        self.pad1_forward = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad1_backward = (0, 0, 0, 0, 0, 0, 1, 0)
        self.pad_second = (0, 0, 0, 0, 0, 0, 1, 1)

        self.conv3 = nn.Conv2d(in_channels=self.channel//self.reduction,
                 out_channels=self.channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.conv3_smallscale2 = nn.Conv2d(in_channels=self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn3_smallscale2 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        
        self.conv3_smallscale4 = nn.Conv2d(in_channels = self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn3_smallscale4 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        ### 二级差 attention 特征

        self.conv_second = nn.Conv2d(in_channels=self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn_second = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_up = nn.Conv1d(in_channels=self.channel//self.reduction,
                                out_channels=self.channel//self.reduction*2,kernel_size=3,stride=1,padding=1,bias=False)

        self.conv_diff = nn.Conv2d(in_channels=self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        ######## 二级差多尺度空间特征 

        self.conv_down_second_spatial = nn.Conv2d(in_channels=self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn_down_second_spatial = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        
        self.conv_origin_second_spatial = nn.Conv2d(in_channels = self.channel//self.reduction,
                                  out_channels=self.channel//self.reduction,padding=1, kernel_size=3, bias=False)
        self.bn_origin_second_spatial = nn.BatchNorm2d(num_features=self.channel//self.reduction)


    def forward(self, x):
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:]) # n, t, c//r, h, w
        
        t_fea_forward, _ = reshape_bottleneck.split([self.n_segment -1, 1], dim=1) # n, t-1, c//r, h, w
       
        
        conv_bottleneck = self.conv2(bottleneck) # nt, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:]) # n, t, c//r, h, w
        _, tPlusone_fea_forward = reshape_conv_bottleneck.split([1, self.n_segment-1], dim=1) # n, t-1, c//r, h, w
        
        
        diff_fea_forward = tPlusone_fea_forward - t_fea_forward # n, t-1, c//r, h, w
        
        
        diff_fea_pluszero_forward = F.pad(diff_fea_forward, self.pad1_forward, mode="constant", value=0) # n, t, c//r, h, w
        diff_fea_pluszero = diff_fea_pluszero_forward
        diff_fea_pluszero_forward = diff_fea_pluszero_forward.view((-1,) + diff_fea_pluszero_forward.size()[2:]) #nt, c//r, h, w
        
    
        ### 一级差conv后，为了制造二级差
        conv_diff_fea_pluszero = self.conv_diff(diff_fea_pluszero_forward)
        conv_diff_fea_pluszero = conv_diff_fea_pluszero.view((-1, self.n_segment) + conv_diff_fea_pluszero.size()[1:]) # n, t, c//r, h, w

        diff_fea_forward, _ = diff_fea_pluszero.split([self.n_segment -1, 1], dim=1) # n, t-1, c//r, h, w
        _, diff_tPlusone_fea_forward = conv_diff_fea_pluszero.split([1, self.n_segment-1], dim=1) # n, t-1, c//r, h, w

        diff_fea_second = diff_tPlusone_fea_forward - diff_fea_forward
        diff_fea_second = diff_fea_second[:,:-1] # 最后一帧二级差是没有意义的，因为最后一帧一级差是0
        diff_fea_second = diff_fea_second.reshape((-1,) + diff_fea_second.size()[2:]) #n6, c//r, h, w

        # diff_fea_second_copy = diff_fea_second
        

        ################ 二级特征差 n,6,c//r,h,w
        # diff_second_pluszero_forward = self.bn_second(self.conv_second(diff_fea_second)) # n6,c//r,h,w
        # diff_second_pluszero_forward = self.avg_pool(diff_second_pluszero_forward) # n6,c//r,1,1

        # ################ 6帧 二级差 上采样为 8帧(其实为7帧)
        # diff_second_pluszero_forward = diff_second_pluszero_forward.reshape(self.n_segment, 6, -1).transpose(1,2) # n,c//r, 6
        # diff_second_pluszero_forward = self.conv_up(diff_second_pluszero_forward) # n,c//r*2,6
        # diff_second_pluszero_forward = diff_second_pluszero_forward.view(self.n_segment,-1,12) # n,c//r,12
        # diff_up0 = diff_second_pluszero_forward[:,:,0].unsqueeze(-1)
        # diff_up1 = (0.5*diff_second_pluszero_forward[:,:,1] + 0.5*diff_second_pluszero_forward[:,:,2]).unsqueeze(-1)
        # diff_up2 = (0.5*diff_second_pluszero_forward[:,:,3] + 0.5*diff_second_pluszero_forward[:,:,4]).unsqueeze(-1)
        # diff_up3 = (0.5*diff_second_pluszero_forward[:,:,5] + 0.5*diff_second_pluszero_forward[:,:,6]).unsqueeze(-1)
        # diff_up4 = (0.5*diff_second_pluszero_forward[:,:,7] + 0.5*diff_second_pluszero_forward[:,:,8]).unsqueeze(-1)
        # diff_up5 = (0.5*diff_second_pluszero_forward[:,:,9] + 0.5*diff_second_pluszero_forward[:,:,10]).unsqueeze(-1)
        # diff_up6 = diff_second_pluszero_forward[:,:,11].unsqueeze(-1)
        # diff_up = torch.cat((diff_up0, diff_up1, diff_up2, diff_up3, diff_up4, diff_up5, diff_up6),dim=2) # n,c//r,7


        # diff_up = diff_up.transpose(1,2).unsqueeze(-1).unsqueeze(-1) # 8,7,8,1,1
        # diff_up = F.pad(diff_up, self.pad1_forward, mode="constant",value=0) # n,8,c//r,1,1
        # diff_up = diff_up.reshape(-1,self.channel//self.reduction,1,1)

        # diff_fea_pluszero_forward = diff_fea_pluszero_forward + diff_fea_pluszero_forward*diff_up

        ######## 对二级差进行多尺度
        diff_fea_second_conv = self.bn_origin_second_spatial(self.conv_origin_second_spatial(diff_fea_second))
        diff_fea_second_down = self.avg_pool_forward2(diff_fea_second)
        diff_fea_second_down = self.bn_down_second_spatial(self.conv_down_second_spatial(diff_fea_second_down))
        diff_fea_second_down = F.interpolate(diff_fea_second_down, diff_fea_second_conv.size()[2:]) # n6,c//r,h,w

        diff_fea_second_fusion = 1/3*diff_fea_second_down + 1/3*diff_fea_second_conv +1/3* diff_fea_second # n6,c//r,h,w

        diff_fea_second_fusion = diff_fea_second_fusion.view((-1, 6) + diff_fea_second_fusion.size()[1:]) # n, 6, c//r,h, w
        diff_fea_second_fusion = F.pad(diff_fea_second_fusion, self.pad_second, mode="constant", value=0) # n, 8, c//r, h, w
        diff_fea_second_fusion = diff_fea_second_fusion.reshape((-1,) + diff_fea_second_fusion.size()[2:]) #n8, c//r, h, w

        diff_fea_second_fusion = self.sigmoid_forward(diff_fea_second_fusion) - 0.5

        # diff_fea_pluszero_forward = diff_fea_pluszero_forward + diff_fea_pluszero_forward*diff_fea_second_fusion


        y_forward_smallscale2 = self.avg_pool_forward2(diff_fea_pluszero_forward) # nt, c//r, 1, 1
        y_forward_smallscale4 = diff_fea_pluszero_forward
      
        y_forward_smallscale2 = self.bn3_smallscale2(self.conv3_smallscale2(y_forward_smallscale2))

        y_forward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(y_forward_smallscale4))

        y_forward_smallscale2 = F.interpolate(y_forward_smallscale2, diff_fea_pluszero_forward.size()[2:])

        
        y_forward = self.bn3(self.conv3(1.0/3.0*diff_fea_pluszero_forward + 1.0/3.0*y_forward_smallscale2 + 1.0/3.0*y_forward_smallscale4))# nt, c, 1, 1
        # y_forward = self.bn3(self.conv3(diff_fea_pluszero_forward))
        y_forward = self.sigmoid_forward(y_forward) - 0.5

        output = x + x*y_forward

        return output


class GloRe(nn.Module):
    def __init__(self, in_channels):
        super(GloRe, self).__init__()
        self.N = in_channels // 4
        self.S = in_channels // 2
        
        self.theta = nn.Conv2d(in_channels, self.N, 1, 1, 0, bias=False) # n,c//4
        self.phi = nn.Conv2d(in_channels, self.S, 1, 1, 0, bias=False)  # n,c//2
        
        self.relu = nn.ReLU()
        
        self.node_conv = nn.Conv1d(self.N, self.N, 1, 1, 0, bias=False) 
        self.channel_conv = nn.Conv1d(self.S, self.S, 1, 1, 0, bias=False)
        
        self.conv_2 = nn.Conv2d(self.S, in_channels, 1, 1, 0, bias=False)
        
    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W
        
        B = self.theta(x).view(-1, self.N, L)

        phi = self.phi(x).view(-1, self.S, L)
        phi = torch.transpose(phi, 1, 2)

        V = torch.bmm(B, phi) / L # (-1, self.N, self.S)
        # V = self.relu(self.node_conv(V)) # (-1, self.N, self.S)
        V_node = self.relu(self.node_conv(V)) # (-1, self.N, self.S)
        V = V - V_node
        ## 这里感觉有点错误，这里是不是应该在用一个V-AgV,然后再送入channel conv
        V = self.relu(self.channel_conv(torch.transpose(V, 1, 2))) # (-1,self.S,self.N)
        
        y = torch.bmm(torch.transpose(B, 1, 2), torch.transpose(V, 1, 2)) # (-1,self.L,self.S)
        y = y.view(-1, self.S, H, W)
        y = self.conv_2(y)  # (N,C,H,W)
        
        return x + y

class MEModule(nn.Module):
    """ Motion exciation module
    
    :param reduction=16
    :param n_segment=8/16
    """
    def __init__(self, channel, reduction=16, n_segment=8):
        super(MEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment

        ############################## Channel attention #########################
        self.reduced_channels = self.channel//16
        self.relu = nn.ReLU(inplace=True)

        self.action_squeeze = nn.Conv2d(self.channel, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, 
                                       groups=1)
        self.action_expand = nn.Conv2d(self.reduced_channels, self.channel, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        
        self.action_bn = nn.BatchNorm2d(self.channel)
        ############################## multiple temporal ##########################

        self.conv8_spatial = nn.Conv2d(self.reduced_channels,self.reduced_channels,3,1,1) # 这里没有下采样，是因为用在了res3和res4，这两个阶段的大小是14*14和7*7，感觉不是很大
        self.conv8_temporal = nn.Conv1d(self.reduced_channels,self.reduced_channels,3,2,1)

        self.conv_second = nn.Conv2d(self.reduced_channels,self.reduced_channels,3,1,1)
        self.conv_second_temporal = nn.Conv1d(self.reduced_channels,self.reduced_channels,3,1,0)

        self.conv1d_upsample = nn.Conv1d(self.reduced_channels,self.reduced_channels*3,1,1)


        ##########################################################################

        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.identity = nn.Identity()

    def forward(self, x):
        nt, c, h, w = x.size()

        ##################channel attention#######################
        ############### 2D convolution: c*T*1*1, channel excitation
        n_batch = nt // self.n_segment
        x_p2 = self.avg_pool(x) # nt,c,h,w ——> nt,c,1,1
        x_p2 = self.action_squeeze(x_p2) # nt,c,1,1 ——> nt,c//16,1,1
        c_r = x_p2.size(1) # c//r
        x_p2 = x_p2.view(n_batch, self.n_segment, c_r, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1) # nt,c//16,1,1 ——> n,c//16,t
        x_p2 = self.action_conv1(x_p2) # n,c//16,t ——> n,c//16,t
        # x_p2 = self.relu(x_p2) 去掉
        x_p2 = x_p2.transpose(1,2).reshape(-1, c_r, 1, 1) # n,c//16,t ——> nt, c//16, 1, 1
        x_p2 = self.action_expand(x_p2) # nt, c//16, 1, 1 ——> nt, c, 1, 1
        x_p2 = self.action_bn(x_p2) # 新加
        x_p2 = self.sigmoid(x_p2) - 0.5 # 减去0.5
        x_p2 = x * x_p2 + x
        ###########################################################

        ########################## multi path #####################

        bottleneck = self.conv1(x_p2) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w

        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
        t_fea, __ = reshape_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:]) # n, t, c//r, h, w
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment-1], dim=1)  # n, t-1, c//r, h, w
        
        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w

        ############### 8帧帧差 n, t, c//r, h, w
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
        
        diff_fea_second01 = (diff_fea_pluszero[:,1,:,:,:] - diff_fea_pluszero[:,0,:,:,:]).unsqueeze(1)
        diff_fea_second12 = (diff_fea_pluszero[:,2,:,:,:] - diff_fea_pluszero[:,1,:,:,:]).unsqueeze(1)
        diff_fea_second23 = (diff_fea_pluszero[:,3,:,:,:] - diff_fea_pluszero[:,2,:,:,:]).unsqueeze(1)
        diff_fea_second34 = (diff_fea_pluszero[:,4,:,:,:] - diff_fea_pluszero[:,3,:,:,:]).unsqueeze(1)
        diff_fea_second45 = (diff_fea_pluszero[:,5,:,:,:] - diff_fea_pluszero[:,4,:,:,:]).unsqueeze(1)
        diff_fea_second56 = (diff_fea_pluszero[:,6,:,:,:] - diff_fea_pluszero[:,5,:,:,:]).unsqueeze(1)

        ############### 二级特征差 n,6,c//r,h,w
        diff_fea_second = torch.cat((diff_fea_second01, diff_fea_second12, diff_fea_second23, diff_fea_second34, diff_fea_second45, diff_fea_second56),dim=1) # n,6,c//r,h,w

        ############### 直接取4帧帧差 n,4,c//r,h,w
        diff4 = torch.cat((diff_fea_pluszero[:,0].unsqueeze(0),diff_fea_pluszero[:,2].unsqueeze(0),diff_fea_pluszero[:,4].unsqueeze(0),diff_fea_pluszero[:,6].unsqueeze(0)),dim=1) # n,4,c//r,h,w
        
        ############## 8帧帧差、4帧直接取出来的帧差、6帧两级差 变换维度
        diff8 = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  #nt, c//r, h, w
        diff4 = diff4.view((-1,) + diff4.size()[2:]) # n4,c//r,h,w
        diff_fea_second = diff_fea_second.view((-1,) + diff_fea_second.size()[2:]) # n6,c//r,h,w

        ################ 8帧变换为4帧
        diff8 = self.conv8_spatial(diff8) # n8,c//r,h,w ——> n8,c//r,h,w
        diff8 = self.avg_pool(diff8).squeeze(-1) # n8,c//r,h,w ——> n8, c//r, 1
        diff8 = diff8.view(n_batch,self.n_segment,-1).transpose(1,2) # n8,c//r,1 ——> n,c//r,8
        diff8 = self.conv8_temporal(diff8) # n,c//r,8 ——> n,c//r,4

        ################ 直接取出来的4帧帧差
        diff4 = self.avg_pool(diff4).squeeze(-1).view(n_batch,-1,c_r).transpose(1,2) # n4,c//r,h,w ——> n,c//r,4

        ################ 6帧二级差
        diff_second = self.conv_second(diff_fea_second) # n6,c//r,h,w
        diff_second = self.avg_pool(diff_second).squeeze(-1) #n6,c//r,1
        diff_second = diff_second.view(n_batch,-1,c_r).transpose(1,2) # n,c//r,6
        diff_second = self.conv_second_temporal(diff_second) # n,c//r,4

        diff4 = 1/3*diff4 + 1/3*diff8 + 1/3*diff_second

        # diff4 = 1/2*diff4 + 1/2*diff8 # n,c//r,4

        ########################################## 如何将4帧恢复出来8帧 #################################

        diff = self.conv1d_upsample(diff4) # n,c//r,4 ——> n,c//r*3,4 
        diff = diff.view(n_batch,c_r,-1) # n,c//r*3,4 ——> n,c//r,3*4
        diff23 = 0.5*diff[:,:,2]+0.5*diff[:,:,3]
        diff23 = diff23.unsqueeze(-1)
        diff56 = 0.5*diff[:,:,5]+0.5*diff[:,:,6]
        diff56 = diff56.unsqueeze(-1)
        diff89 = 0.5*diff[:,:,8]+0.5*diff[:,:,9]
        diff89 = diff89.unsqueeze(-1)
        diff = torch.cat((diff[:,:,1].unsqueeze(-1), diff23, diff[:,:,4].unsqueeze(-1), diff56, diff[:,:,7].unsqueeze(-1), diff89,diff[:,:,10:12]),dim=2) # n,c//r,3*4  ——> n,c//r,8

        #diff = diff.view(n_batch,c_r,-1).contiguous().view(n_batch*self.n_segment,c_r,1,1) # n,c//r*2,4 ——> n8,c//r,1,1

        diff = diff.transpose(1,2).reshape(-1,c_r,1,1) # n,c//r,8 ——> n8,c//r,1,1
        #y = self.avg_pool(diff)  # nt, c//r, 1, 1
        #y = self.conv3(y)  # nt, c, 1, 1
        y = self.conv3(diff)
        y = self.bn3(y)  # nt, c, 1, 1
        y = self.sigmoid(y)  # nt, c, 1, 1
        y = y - 0.5
        output = x_p2 + x_p2 * y.expand_as(x_p2)
        return output

class ShiftModule(nn.Module):
    def __init__(self, input_channels, n_segment=8,n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.divded = self.input_channels % self.fold_div
        if self.divded:
            self.conv = nn.Conv1d(self.fold_div*self.fold+self.divded, self.fold_div*self.fold+self.divded,
                kernel_size=3, padding=1, groups=self.fold_div*self.fold+self.divded,
                bias=False)
        else:
            self.conv = nn.Conv1d(self.fold_div*self.fold, self.fold_div*self.fold,
                kernel_size=3, padding=1, groups=self.fold_div*self.fold,
                bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1) # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        x = self.conv(x) # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2) # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleneckShift(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(BottleneckShift, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.num_segments = num_segments
        # self.mse = mSEModule(planes, n_segment=self.num_segments,index=1) #
        # self.mse = mSEModule_second(planes, n_segment=self.num_segments,index=1) # 
        #self.glore = GloRe(planes)
        # self.mem = MEModule(planes)

        self.conv_t = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=1,
            bias=False)
        self.bn_t = nn.BatchNorm2d(num_features=planes)
        self.sigmoid = nn.Sigmoid()


        self.conv_all = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=1,
            bias=False)
        self.bn_all = nn.BatchNorm2d(num_features=planes)
        self.sigmoid_all = nn.Sigmoid()


        self.adaptive_token = TokenLearnerModule(num_tokens=10, dim=planes)
        self.TokenLearnerModule1D = TokenLearnerModule1D(num_tokens=4, dim=planes)

        self.spatial_atten = FSAttention(dim=planes, heads=8, dropout=0., hidden_dim=128)
        self.temporal_atten = TemporalAttention(dim=planes, heads=8, dropout=0., hidden_dim=128)

        self.shift = ShiftModule(planes, n_segment=self.num_segments, n_div=8, mode='shift')
      

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
    
    def creat_pe_absolute_sincos_embedding(self, n_pos_vec, dim):  # n(toke number), dim
        assert dim % 2 == 0, "wrong dim"
        position_embedding = torch.zeros(n_pos_vec.numel(), dim, dtype=torch.float)

        omega = torch.arange(dim//2, dtype=torch.float)
        omega /= dim/2.
        omega = 1./(10000**omega)

        sita = n_pos_vec[:,None] @ omega[None,:]
        emb_sin = torch.sin(sita)
        emb_cos = torch.cos(sita)

        position_embedding[:,0::2] = emb_sin
        position_embedding[:,1::2] = emb_cos

        return position_embedding
    # 用之前需要n_pos_vec = torch.arange(8, dtype=torch.float)后再传入
    

    def get_positional_encoding(self, H, W, dim=128):
        """
        2D positional encoding, following
            https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
        """

        assert dim % 4 == 0, 'dim must be a multiply of 4 (h/w x sin/cos)'
        c_period = 10000. ** torch.linspace(0., 1., dim // 4)
        h_vec = torch.arange(0, H).reshape((H, 1, 1)).repeat(1, W, 1) / c_period
        w_vec = torch.arange(0, W).reshape((1, W, 1)).repeat(H, 1, 1) / c_period
        position_encoding = torch.cat(
            (torch.sin(h_vec), torch.cos(h_vec), torch.sin(w_vec), torch.cos(w_vec)), dim=-1)
        position_encoding = position_encoding.reshape((1, H, W, dim))
        
        return position_encoding # 1, H, W, 128


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        
        # out = self.mse(out)
        # out = self.glore(out)
        # out = self.mem(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out_ori = out

        c_dim, height_, width_ = out.size()[1], out.size()[2], out.size()[3]

        spatial_pos = self.get_positional_encoding(height_, width_, c_dim).permute(0, 3, 1, 2).cuda() # bt=1, c, h, w
        temporal_pos = self.creat_pe_absolute_sincos_embedding(torch.arange(self.num_segments, dtype=torch.float), c_dim).cuda() # n*dim
        # pos = nn.Parameter(torch.randn(1, 1, height_, width_)).cuda()
        temporal_pos = temporal_pos.reshape(self.num_segments, c_dim).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # 1, t, c, 1, 1
        out = out + spatial_pos # bt, c, h, w
        out = out.reshape(-1, self.num_segments, c_dim, height_, width_)

        out = out + temporal_pos
        out = out.reshape(-1, c_dim, height_, width_)
         

        # 长时模块嵌入
        num1_token = self.adaptive_token(out) # out:bt, c, h, w  out:bt, num1_token, c
        num2_token = self.TokenLearnerModule1D(num1_token) # bt, num2_token, c
        # print(num1_token.size(), num2_token.size()) # [64, 10, 128] [64, 4, 128]
        out = self.spatial_atten(out, num1_token, num1_token, num2_token, num2_token) # bt, c, h//4, w//4
        # print('temporal: ', temporal_feat.size()) # 8, 8, 128 
        # print(out.shape) # 64, 128, 7, 7
        
        temporal_feat = self.temporal_atten(self.num_segments, num1_token, num1_token) # b, t, c
        temporal_feat = temporal_feat.reshape(-1, c_dim, 1, 1)
        temporal_feat = self.conv_t(temporal_feat)
        temporal_feat = self.bn_t(temporal_feat)
        temporal_feat = self.sigmoid(temporal_feat)


        # out = out.reshape(temporal_feat.shape[0], temporal_feat.shape[1], out.shape[1], out.shape[2], out.shape[3])
        # temporal_feat = temporal_feat.unsqueeze(-1).unsqueeze(-1)

        out = out + out*temporal_feat # bt, c, h//4, w//4
        # out = out.reshape(-1, out.shape[2], out.shape[3], out.shape[4]) # bt, c, h//4, w//4
        out = F.interpolate(out, [height_, width_])
        out = self.conv_all(out)
        out = self.bn_all(out)
        out = self.sigmoid_all(out)

        out = out_ori + out_ori*out


        out = self.shift(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HSBottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, downsample=None, stride=1, s=4):
        super(HSBottleNeck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.num_segments = num_segments # 自己的程序设计也是做了压缩的，不确定是否会与这个shift产生冲突
        self.shift = ShiftModule(planes, n_segment=self.num_segments, n_div=8, mode='shift')

        # 特征图尺度改变，用原生ResBlock
        if stride != 1:
            # self.conv_3x3 = nn.Sequential(
            #     nn.Conv2d(planes, planes, stride=stride, kernel_size=3, padding=1, bias=False),
            #     nn.BatchNorm2d(planes),
            #     nn.ReLU(inplace=True),)
            # out_ch = out_channels
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=True)
            self.bn2 = nn.BatchNorm2d(planes)
        # 特征图尺度不变，用HSBlock
        else:
            self.conv_3x3 = HSBlock(in_ch=planes, s=s)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #x_hs = self.conv_1x1(x)
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.shift(out)

        if self.stride != 1:
            out = self.conv2(out)
            out = self.bn2(out)
        else:
            out = self.conv_3x3(out)
            
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out

# class HSBottleNeck(nn.Module):
#     """Residual block for resnet over 50 layers
#     """
#     expansion = 4

#     def __init__(self, num_segments, inplanes, planes, downsample=None, stride=1, s=8):
#         super(HSBottleNeck,self).__init__()
        
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.num_segments = num_segments # 自己的程序设计也是做了压缩的，不确定是否会与这个shift产生冲突
#         #self.shift = ShiftModule(planes, n_segment=self.num_segments, n_div=8, mode='shift')

#         # 特征图尺度改变，用原生ResBlock
#         if stride != 1:
#             self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=True)
#             self.bn2 = nn.BatchNorm2d(planes)
#         # 特征图尺度不变，用HSBlock
#         else:
#             self.conv_squeeze = nn.Conv2d(planes,planes//8,kernel_size=1,bias=True)
#             self.conv_3x3 = HSBlock(in_ch=planes, s=s)
#             self.conv_unsqueeze = nn.Conv2d(planes//8,planes,kernel_size=1,bias=True)

#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

        

#     def forward(self, x):
#         ## new idea: 将特征压缩为原来的1/8，然后变为N,TC,H,W,再针对TC分为8份，再利用HSblock，之后再恢复特征原样
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         #out = self.shift(out) 先暂时不加，还没想明白

#         if self.stride != 1:
#             out = self.conv2(out)
#             out = self.bn2(out)
#         else:
#             N,C,H,W  = out.size()
#             out = self.conv_squeeze(out) # N,C//8,H,W
#             x_c = out.size(1)
#             out = out.view(N//self.num_segments,self.num_segments*x_c,H,W) # B,C*num_seg,H,W
#             out = self.conv_3x3(out) # B,C*num_seg,H,W
#             x_c = out.size(1)
#             out = out.view(N,x_c//self.num_segments,H,W) # N,C//8,H,W
#             out = self.conv_unsqueeze(out) # N,C,H,W

#             ###################
#             #out = self.conv_3x3(out)
#             ###################
            

#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)
        
#         return out

class Bottle2neckShift(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neckShift, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        width_avg = copy.deepcopy(width)
        self.num_segments = 8 ## 这个值其实不应该写死，后期应该会变

        #self.me = MEModule(width*scale, reduction=16, n_segment=8)

        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)

        self.mse = mSEModule(width*scale, n_segment=self.num_segments,index=1)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        shifts = []
        acc_width = 0
        for i in range(self.nums):
            ############### 修改 ##############  只有stride==1,才执行分层
            if stride==1:
                if i == 0:
                    acc_width = width//2
                elif i == self.nums - 1:
                    width = width_avg + acc_width
                else:
                    width=width+acc_width
                    acc_width= ceil(width/2)
            ############## 修改 ################
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride,padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
            shifts.append(ShiftModule(width, n_segment=8, n_div=2, mode='fixed'))
        #shifts.append(ShiftModule(width, n_segment=8, n_div=2, mode='shift'))源码
        shifts.append(ShiftModule(width_avg, n_segment=8, n_div=2, mode='shift'))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.shifts = nn.ModuleList(shifts)

        # self.conv3 = nn.Conv2d(width*scale, planes * self.expansion,
        #            kernel_size=1, bias=False)源码
        self.conv3 = nn.Conv2d(width_avg*scale, planes * self.expansion,
                   kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        #self.width = width源码
        self.width = width_avg
        self.stride = stride

    def forward(self, x):
        # import pdb; pdb.set_trace()
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.mse(out)

        #out = self.me(out)
        ######################## 修改 ################################
        spx = torch.split(out, self.width, 1)  # 4*(nt, c/4, h, w)
        for i in range(self.nums):
            if i == 0 or (self.stype == 'stage' and self.stride==2):
                sp = spx[i]
            else:
                sp = torch.cat((spx[i],sp1), 1)
            sp = self.shifts[i](sp)
            sp = self.convs[i](sp) # 含有stride，stride=2会改变大小
            sp = self.relu(self.bns[i](sp))
            sp1,sp2 = sp.chunk(chunks=2,dim=1) # 39分为2份，sp1为20，sp2为19
            if self.stride==1:
                if i == 0:
                    out = sp1
                if i==1:
                    out = torch.cat((out, sp2), 1)
                if i == self.nums -1:
                    out = torch.cat((out,sp),1)
            if self.stype=='stage' and self.stride==2:
                if i == 0:
                    out = sp
                else:
                    out = torch.cat((out, sp), 1)
        last_sp = spx[self.nums]
        last_sp = self.shifts[self.nums](last_sp)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, last_sp), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(last_sp)), 1)
        ##############################################################
        
        ######################### 源码 ###############################
        # spx = torch.split(out, self.width, 1)  # 4*(nt, c/4, h, w)
        # for i in range(self.nums):
        #     if i == 0 or self.stype == 'stage':
        #         sp = spx[i]
        #     else:
        #         sp = sp + spx[i]
        #     sp = self.shifts[i](sp)
        #     sp = self.convs[i](sp)
        #     sp = self.relu(self.bns[i](sp))
        #     if i == 0:
        #         out = sp
        #     else:
        #         out = torch.cat((out, sp), 1)
        # last_sp = spx[self.nums]
        # last_sp = self.shifts[self.nums](last_sp)
        # if self.scale != 1 and self.stype == 'normal':
        #     out = torch.cat((out, last_sp), 1)
        # elif self.scale != 1 and self.stype == 'stage':
        #     out = torch.cat((out, self.pool(last_sp)), 1)
        #############################################################

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FBResNet(nn.Module):

    def __init__(self, num_segments, block, layers, num_classes=1000):
        self.inplanes = 64

        self.input_space = None
        self.input_size = (224, 224, 3)
        self.mean = None
        self.std = None
        self.num_segments = num_segments
        super(FBResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.num_segments,Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(self.num_segments,block, 128, layers[1], stride=2)
        # self.layer2 = self._make_layer_hs(self.num_segments,HSBottleNeck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.num_segments,block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.num_segments,block, 512, layers[3], stride=2)
        # self.layer3 = self._make_layer_hs(self.num_segments,HSBottleNeck, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(self.num_segments,block, 512, layers[3], stride=2)
        #self.layer4 = self._make_layer_hs(self.num_segments,HSBottleNeck, 512, layers[3], stride=2)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, num_segments ,block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(num_segments, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(num_segments, self.inplanes, planes)) #第一个block是比较特殊的，stride要么是1，要么是2

        return nn.Sequential(*layers)
    
    def _make_layer_hs(self, num_segments ,block, planes, blocks, stride=1,s=4):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(num_segments, self.inplanes, planes, downsample, stride, s))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(num_segments, self.inplanes, planes)) #第一个block是比较特殊的，stride要么是1，要么是2

        return nn.Sequential(*layers)

   


    def features(self, input):
        #print('input size: ',input.size())
        x = self.conv1(input)
        #self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
       
        x = self.layer2(x)
       
        x = self.layer3(x)
        
        x = self.layer4(x)
       
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

class Res2Net(nn.Module):
    
    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
            stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def fbres2net50(pretrained=False, **kwargs):
    """Constructs a TEA model.
    part of the TEA model refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #path = '/home/yckj3949/sheng/res2net50_26w_4s-06e79181.pth' #yuncong
    path = '/home/sheng/sheng/res2net50_26w_4s-06e79181.pth' # ps
    model_pth = torch.load(path, map_location='cpu')
    model_res2net = Res2Net(Bottle2neckShift, [3, 4, 6, 3], baseWidth = 26, scale = 4)
    model_dict= model_res2net.state_dict()
    if pretrained:
        for key1,value1 in model_pth.items():
            if key1 in model_res2net.state_dict():
                if 'conv' in key1:
                    if model_pth[key1].size()!= model_dict[key1].size():
                        #################   求均值     #################
                        # params = model_pth[key1]
                        # old_kernel_size = params.size()
                        # kernel_size = model_res2net.state_dict()[key2].size()
                        # kernel_size_temp = old_kernel_size[:1] + kernel_size[1:]
                        
                        # new_kernels_temp = params.mean(dim=1, keepdim=True).expand(kernel_size_temp).contiguous()
                        # new_kernels = new_kernels_temp.mean(dim=0, keepdim=True).expand(kernel_size).contiguous()
                        # model_res2net.state_dict()[key2] = new_kernels
                        ###################################################

                        ##################    拷贝和concat  #########################
                        params = model_pth[key1]
                        old_kernel_size = params.size() #[26,26,3,3]
                        kernel_size = model_dict[key1].size() #[39,39,3,3]
                        delta = kernel_size[0] - old_kernel_size[0]
                        new_kernels_temp = torch.cat((params,params[:delta]),dim=0)
                        new_kernels = torch.cat((new_kernels_temp,new_kernels_temp[:,:delta,:,:]),dim=1)
                        
                        model_pth_key1 = {key1:new_kernels}
                        pretrained_dict={k:v for k,v in model_pth_key1.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        model_res2net.load_state_dict(model_dict)

                    else:
                        model_pth_key1 = {key1:model_pth[key1]}
                        pretrained_dict={k:v for k,v in model_pth_key1.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        model_res2net.load_state_dict(model_dict)

                if 'bn' in key1:
                    if model_pth[key1].size()!= model_dict[key1].size():
                        params = model_pth[key1]
                        old_bn_size = params.size()
                        bn_size = model_dict[key1].size()
                        delta = bn_size[0] - old_bn_size[0]
                        new_bn = torch.cat((params,params[:delta]))

                        model_pth_key1 = {key1:new_bn}
                        pretrained_dict={k:v for k,v in model_pth_key1.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        model_res2net.load_state_dict(model_dict)

                    else:
                        model_pth_key1 = {key1:model_pth[key1]}
                        pretrained_dict={k:v for k,v in model_pth_key1.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        model_res2net.load_state_dict(model_dict)
                        
    return model_res2net


def fbresnet50(num_segments=8,pretrained=False,num_classes=1000):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(num_segments,BottleneckShift, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['fbresnet50']),strict=False)
        path_root  = os.path.dirname(os.path.abspath(__file__))
        path_pth = os.path.join(path_root,'resnet50-19c8e357.pth')
        model.load_state_dict(torch.load(path_pth,map_location='cpu'),strict=False)
         
    return model


def fbresnet101(num_segments,pretrained=False,num_classes=1000):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(num_segments,BottleneckShift, [3, 4, 23, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['fbresnet101']),strict=False)
    return model

if __name__=='__main__':
  
    import torch
    
    # for key,value in model_pth.items():
    #     #print(key,value.size(),sep="   ")
    #     print(model_pth[key]) #已经是tensor
    
    #for key,value in model_res2net.state_dict().items():
        #print(key,value.shape)
        #print(model_res2net.state_dict()[key]) # 已经是tensor

    model_pth = torch.load('/home/sheng/.cache/torch/hub/checkpoints/res2net50_26w_4s-06e79181.pth')
    model_res2net = Res2Net(Bottle2neckShift, [3, 4, 6, 3], baseWidth = 26, scale = 4)
    model_dict= model_res2net.state_dict()
    for key1,value1 in model_pth.items():
        if key1 in model_res2net.state_dict():
            if 'conv' in key1:
                if model_pth[key1].size()!= model_dict[key1].size():
                    #################   求均值     #################
                    # params = model_pth[key1]
                    # old_kernel_size = params.size()
                    # kernel_size = model_res2net.state_dict()[key2].size()
                    # kernel_size_temp = old_kernel_size[:1] + kernel_size[1:]
                    
                    # new_kernels_temp = params.mean(dim=1, keepdim=True).expand(kernel_size_temp).contiguous()
                    # new_kernels = new_kernels_temp.mean(dim=0, keepdim=True).expand(kernel_size).contiguous()
                    # model_res2net.state_dict()[key2] = new_kernels
                    ###################################################

                    ##################    拷贝和concat  #########################
                    params = model_pth[key1]
                    old_kernel_size = params.size() #[26,26,3,3]
                    kernel_size = model_dict[key1].size() #[39,39,3,3]
                    delta = kernel_size[0] - old_kernel_size[0]
                    new_kernels_temp = torch.cat((params,params[:delta]),dim=0)
                    new_kernels = torch.cat((new_kernels_temp,new_kernels_temp[:,:delta,:,:]),dim=1)
                    
                    model_pth_key1 = {key1:new_kernels}
                    pretrained_dict={k:v for k,v in model_pth_key1.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    model_res2net.load_state_dict(model_dict)

                else:
                    
                    model_pth_key1 = {key1:model_pth[key1]}
                    pretrained_dict={k:v for k,v in model_pth_key1.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    model_res2net.load_state_dict(model_dict)

            if 'bn' in key1:
                if model_pth[key1].size()!= model_dict[key1].size():
                    params = model_pth[key1]
                    old_bn_size = params.size()
                    bn_size = model_dict[key1].size()
                    delta = bn_size[0] - old_bn_size[0]
                    new_bn = torch.cat((params,params[:delta]))

                    model_pth_key1 = {key1:new_bn}
                    pretrained_dict={k:v for k,v in model_pth_key1.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    model_res2net.load_state_dict(model_dict)

                else:
                    model_pth_key1 = {key1:model_pth[key1]}
                    pretrained_dict={k:v for k,v in model_pth_key1.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    model_res2net.load_state_dict(model_dict)
    









    