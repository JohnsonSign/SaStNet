# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
from ops.tdn_net import tdn_net
import torch.nn.functional as F

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,crop_num=1,
                 partial_bn=True, print_spec=True, pretrain='imagenet',fc_lr5=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5  # fine_tuning for UCF/HMDB
        self.target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}

        self.relu = nn.ReLU()

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model, self.num_segments)
        feature_dim = self._prepare_tsn(num_class)
        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        
        #print(self.base_model)
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        
        return feature_dim

    def _prepare_base_model(self, base_model, num_segments):
        print(('=> base model: {}'.format(base_model)))
        if 'resnet' in base_model :
            self.base_model = tdn_net(base_model, num_segments)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
        else :
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []
        inorm = []
        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.LayerNorm):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        if self.fc_lr5: # fine_tuning for UCF/HMDB
            return [
                {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
                'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
                'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                'name': "BN scale/shift"},
                {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
                'name': "custom_ops"},
                {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
                'name': "lr5_weight"},
                {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
                'name': "lr10_bias"},
            ]
        else : # default 
            return [
                {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
                'name': "first_conv_weight"},
                {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
                'name': "first_conv_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "normal_bias"},
                {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
                'name': "BN scale/shift"},
                {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
                'name': "custom_ops"},
            ]



    def forward(self, input, no_reshape=False):
        if not no_reshape: # 走这里
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
            # print('input size: ', input.size()) # [8, 120, 224, 224]
            # print('input size: ', input.view((-1, sample_len*5) + input.size()[-2:]).size()) [64, 15, 224, 224]
            base_out = self.base_model(input.view((-1, sample_len*5) + input.size()[-2:])) # 走这里
        else:
            base_out = self.base_model(input)

        if self.dropout > 0: # dropout=0.5 
            base_out = self.new_fc(base_out)
            # print(self.new_fc.weight.shape) # 174, 2048
            # print(base_out.shape) # 64, 174

            # # 
            # fc_weights = self.new_fc.weight.detach().transpose(0, 1).unsqueeze(0).unsqueeze(1) # 1, 1, 2048, 174
            # fc_weights = fc_weights.repeat(target.shape[0], self.num_segments, 1, 1) # 8, 8, 2048, 174

            # # 
            # x_last_feat = x_last_feat.detach().reshape(target.shape[0], self.num_segments, 7, 7, 2048)
            
            # cam = []
            # for i in range(len(target)):
            #     fc_weight_each = fc_weights[i,:,:,target[i]].unsqueeze(-2).unsqueeze(-2) # 8, 1,1, 2048
            #     # print(fc_weight_each.shape)
            #     # print(torch.sum(fc_weight_each * x_last_feat[i,:,:,:,:], dim=-1).shape) # [8,7,7]
            #     cam.append(torch.sum(fc_weight_each * x_last_feat[i,:,:,:,:], dim=-1))  # [8,7,7,2048]    [8,7,7]
            # cam = torch.stack(cam, dim=0) # 8,8,7,7
            # cam = cam.reshape(-1, 7, 7).unsqueeze(1) # 64, 1, 7, 7
            # cam = self.relu(cam)
            # cam -= torch.min(cam)
            # cam /= torch.max(cam) - torch.min(cam)
            # # print(cam)
            # cam  = F.interpolate(cam,[56, 56], mode='bilinear') # 64, 1, 56, 56


        if not self.before_softmax: # 不走这里
            base_out = self.softmax(base_out)

        if self.reshape: # 走这里
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            output = output.squeeze(1)
            # print('output shape: ', output.shape) # 8, 174
            return output


    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
        else:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                            GroupRandomHorizontalFlip_sth(self.target_transforms),
                                            ])

