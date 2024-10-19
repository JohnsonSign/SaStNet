# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from ops.base_module import *
# from torchvision import utils as torch_utils

class TDN_Net(nn.Module):

    def __init__(self,resnet_model,resnet_model1,apha,belta):
        super(TDN_Net, self).__init__()

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)

        # implement conv1_5 and inflate weight 
        self.conv1_temp = list(resnet_model1.children())[0]
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size() 
        new_kernel_size = kernel_size[:1] + (3 * 4,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
    
        
        self.conv1_5 = nn.Sequential(nn.Conv2d(12,64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv1_5[0].weight.data = new_kernels

        # self.conv1_4 = nn.Sequential(nn.Conv2d(9,64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        # new_kernel_size4 = kernel_size[:1] + (3*3, ) + kernel_size[2:]
        # new_kernel4 = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size4).contiguous()
        # self.conv1_4[0].weight.data = new_kernel4

        # self.conv1_2 = nn.Sequential(nn.Conv2d(6,64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        # new_kernel_size2 = kernel_size[:1] + (2*3,) + kernel_size[2:]
        # new_kernel2 = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size2).contiguous()
        # self.conv1_2[0].weight.data = new_kernel2


        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnext_layer1 =nn.Sequential(*list(resnet_model1.children())[4])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avg_diff = nn.AvgPool2d(kernel_size=2,stride=2)
        #self.fc = list(resnet_model.children())[8]
        self.fc = list(resnet_model.children())[-1]
        self.apha = apha
        self.belta = belta

        self.salient = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 1, kernel_size=3, padding=1, bias=False),
                        nn.Sigmoid(),
                        )
        # self.avgpool56 = nn.AvgPool2d(56, stride=1)
        # self.fcres2 = nn.Linear(256, 174, bias=False)
        # self.salient_cam = nn.Sequential(
        #                 nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
        #                 nn.BatchNorm2d(1),
        #                 nn.ReLU(),
        #                 nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
        #                 nn.Sigmoid(),
        #                 )
      

    def forward(self, x):
        x1, x2, x3, x4, x5 = x[:,0:3,:,:], x[:,3:6,:,:], x[:,6:9,:,:], x[:,9:12,:,:], x[:,12:15,:,:]
        #x_c5 = self.conv1_5(self.avg_diff(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],1).view(-1,12,x2.size()[2],x2.size()[3])))
        #print(x2-x1) # [-1,1]之间
        #mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        delta1 = x2-x1 # [64, 3, 224, 224]
        delta2 = x3-x2
        delta3 = x4-x3
        delta4 = x5-x4
       
        ############################v3:这个版本新写的############################
        x = self.conv1(x3)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 56
        
        delta11 = torch.cat([delta1, delta2, delta3, delta4],1).view(-1,12,x2.size()[2],x2.size()[3]) # 224
       
        delta11 = self.avg_diff(delta11) # 112
        delta11 = self.conv1_5(delta11) # 56
        delta11 = self.maxpool_diff(delta11) # 28
        
        delta = delta11 
        x_diff = delta

        delta = F.interpolate(delta,x.size()[2:]) # 56
        x = self.apha*x + self.belta*delta # 56


        x_diff = self.resnext_layer1(x_diff) #28,28
        x = self.layer1_bak(x) # 64,256,56,56
        x_diff = F.interpolate(x_diff, x.size()[2:])
        x = self.apha*x + self.belta*x_diff

        ###########################################################
        salient = self.salient(x) # 64,256,56,56

        # 64,1,56,56

        # salient_cls = self.avgpool56(salient).squeeze(-1).squeeze(-1) # 64,256
        # salient_cls = self.fcres2(salient_cls)
        # salient_cls = salient_cls.reshape(8, 8, 174)
        # salient_cls = salient_cls.mean(dim=1, keepdim=False) # 8,174


        # fc_weights = self.fcres2.weight.detach() # 174, 256
        # x_last_feat = salient.detach().reshape(8, 8, 256, 56, 56)

        # cam = []
        # for i in range(8):
        #     fc_weight_each = fc_weights[target[i], :].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # 1,256,1,1
        #     # print(fc_weight_each.shape)
        #     # print(torch.sum(fc_weight_each * x_last_feat[i,:,:,:,:], dim=-1).shape) # 8,56,56
        #     cam.append(torch.sum(fc_weight_each * x_last_feat[i,:,:,:,:], dim=1))  # 8,256,56,56  -> 8,56,56
        # cam = torch.stack(cam, dim=0) # 8,8,56,56
        # cam = cam.reshape(-1, 56, 56).unsqueeze(1) # 64,1,56,56
        # cam = self.relu(cam)
        # cam -= torch.min(cam)
        # cam /= torch.max(cam) - torch.min(cam)
        # print(cam)

        # cam = self.salient_cam(cam)

        x = x + x*salient
        #############################################################

        x = self.layer2_bak(x) # 64,512,28,28
        x = self.layer3_bak(x) # 64,1024,14,14
        x = self.layer4_bak(x) # 64,2048,7,7

        # x_last_feat = x # 64,2048,7,7
        # x_last_feat = x_last_feat.permute(0, 2, 3, 1) # 64,7,7,2048

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    # #@classmethod
    # def motion_diff(self,x1,x2,x3,x4,x5):
    #     ## 处理short motion
    #     ## 自己想法：
    #     ## 1. 先卷积motion diff
    #     ## 2. 对每一个motion diff conv进行通道split，然后转置，
    #     ##对于每一列作为一个新的motion（为了保证连续性，可以进行一次新的conv）
    #     ## 3. 对于这些conv，利用hierical分层联合起来这些motion信息
    #     ## 4. 利用最后输出的这个motion信息进行抠maks和后续的处理。
    #     delta1 = self.maxpool_diff(self.conv1(x2-x1))
    #     delta2 = self.maxpool_diff(self.conv1_2(x3-x2))
    #     delta3 = self.maxpool_diff(self.conv1_3(x4-x3))
    #     delat4 = self.maxpool_diff(self.conv1_4(x5-x3))

    #     group1 = torch.split(delta1,16,dim=1)
    #     group2 = torch.split(delta2,16,dim=1)
    #     group3 = torch.split(delta3,16,dim=1)
    #     group4 = torch.split(delat4,16,dim=1)

    #     group1_trans = self.conv_group1(torch.cat((group1[0],group2[0],group3[0],group4[0]),dim=1)) #torch.Size([8, 64, 56, 56])
    #     group2_trans = self.conv_group2(torch.cat((group1[1],group2[1],group3[1],group4[1]),dim=1))
    #     group3_trans = self.conv_group3(torch.cat((group1[2],group2[2],group3[2],group4[2]),dim=1))
    #     group4_trans = self.conv_group4(torch.cat((group1[3],group2[3],group3[3],group4[3]),dim=1))

    #     x_trans = torch.cat((group1_trans,group2_trans,group3_trans,group4_trans),dim=1) #torch.Size([8, 64, 56, 56])
        
    #     x = self.hierar(x_trans)
    #     x = x + x_trans

    #     return x
   
    # def motion_importance():
    #     ## 处理motion 片段重要性，加权整个视频帧中的重要性
    #     pass


def tdn_net(base_model=None,num_segments=8,pretrained=True, **kwargs):
    if("50" in base_model):
        resnet_model = fbresnet50(num_segments, pretrained)
        resnet_model1 = fbresnet50(num_segments, pretrained)
        # resnet_model = fbres2net50(pretrained)
        # resnet_model1 = fbres2net50(pretrained)
    else:
        resnet_model = fbresnet101(num_segments, pretrained)
        resnet_model1 = fbresnet101(num_segments, pretrained)

    if(num_segments is 8):
        model = TDN_Net(resnet_model,resnet_model1,apha=0.5,belta=0.5)
    else:
        model = TDN_Net(resnet_model,resnet_model1,apha=0.75,belta=0.25)
    return model

# class Bottle2neck(nn.Module):
#     #expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=16, scale = 4, stype='normal'):
#         """ Constructor
#         Args:
#             inplanes: input channel dimensionality
#             planes: output channel dimensionality
#             stride: conv stride. Replaces pooling layer.
#             downsample: None when stride = 1
#             baseWidth: basic width of conv3x3,源码设置为26
#             scale: number of scale.
#             type: 'normal': normal set. 'stage': first block of a new stage.
#         """
#         super(Bottle2neck, self).__init__()

#         width = int(math.floor(planes * (baseWidth/64.0)))
#         self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(width*scale)
        
#         if scale == 1:
#           self.nums = 1
#         else:
#           #self.nums = scale - 1
#           self.nums = scale
#         if stype == 'stage':
#             self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
#         convs = []
#         bns = []
#         for i in range(self.nums):
#           convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
#           bns.append(nn.BatchNorm2d(width))
#         self.convs = nn.ModuleList(convs)
#         self.bns = nn.ModuleList(bns)

#         #self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False) 源码
#         #self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.conv3 = nn.Conv2d(width*scale,inplanes,kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(inplanes)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stype = stype
#         self.scale = scale
#         self.width  = width

#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         #         m.weight.data.normal_(0, math.sqrt(2. / n))
#         #     elif isinstance(m, nn.BatchNorm2d):
#         #         m.weight.data.fill_(1)
#         #         m.bias.data.zero_()

#     def forward(self, x):
#         # import pdb; pdb.set_trace()
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         #print('out1 size: ',out.size()) # torch.Size([64, 104, 56, 56])

#         spx = torch.split(out, self.width, dim=1)
#         for i in range(self.nums):
#           if i==0 or self.stype=='stage':
#             sp = spx[i]
#           else:
#             sp = sp + spx[i]
#           sp = self.convs[i](sp)
#           sp = self.relu(self.bns[i](sp))
#           if i==0:
#             out = sp
#           else:
#             out = torch.cat((out, sp), 1)
#         # if self.scale != 1 and self.stype=='normal':
#         #   out = torch.cat((out, spx[self.nums]),1)
#         # elif self.scale != 1 and self.stype=='stage':
#         #   out = torch.cat((out, self.pool(spx[self.nums])),1)
        
#         #print('out 2 size: ',out.size())

#         out = self.conv3(out)
#         #print('out 3 size: ',out.size())

#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)
#         # print('residual size: ',residual.size()) #torch.Size([64, 64, 56, 56])
#         # print('out size: ',out.size()) #torch.Size([64, 256, 56, 56])
#         out += residual
#         out = self.relu(out)

#         return out


if __name__=='__main__':
    images = torch.rand(64, 64, 56, 56)
    model = Bottle2neck(64,64)
    output = model(images)
    print(output.size())
    # x1 = torch.randn((8,3,224,224))
    # x2 = torch.randn((8,3,224,224))
    # x3 = torch.randn((8,3,224,224))
    # x4 = torch.randn((8,3,224,224))
    # x5 = torch.randn((8,3,224,224))
    # resnet_model = fbresnet50(num_segments=8, pretrained=True)
    # resnet_model1 = fbresnet50(num_segments=8, pretrained=True)
    # tdn = TDN_Net(resnet_model,resnet_model1,0.5,0.5)
    # out = tdn.motion_diff(x1,x2,x3,x4,x5)
    # model = Bottle2neck(64,64)
    # output = model(out)
    # print(output.size())