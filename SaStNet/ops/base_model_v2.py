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
import numpy as np 
from einops import rearrange
from torch import einsum


class TokenLearnerModule(nn.Module):
    def __init__(self, num_tokens, dim):
        super(TokenLearnerModule, self).__init__()
        self.num_tokens = num_tokens

        self.norm = nn.BatchNorm2d(dim)
        self.atten = nn.Sequential(
                    nn.Conv2d(dim, self.num_tokens, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.num_tokens),
                    nn.ReLU(),
                    nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.num_tokens),
                    nn.ReLU(),
                    nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.num_tokens),
                    nn.ReLU(),
                    nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=3, padding=1, bias=False),
                    nn.Sigmoid())

    def forward(self, inputs): # [B, C, H, W]

        feat = self.norm(inputs)
        feat = self.atten(feat)
        b, num_token, h, w = feat.size()
        feat = feat.reshape((b, num_token, h*w, 1))                   # b, n_token, h*w, 1

        feat_inputs = inputs
        feat_inputs = feat_inputs.reshape((b, -1, h*w))
        feat_inputs = torch.transpose(feat_inputs, 1, 2).unsqueeze(1)  # b, 1, h*w, c
        feat_inputs = torch.sum(feat * feat_inputs, axis=2)            # b, num_token, c

        return feat_inputs



class TokenLearnerModule1D(nn.Module):
    def __init__(self, num_tokens, dim):
        super(TokenLearnerModule1D, self).__init__()
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(dim)
        self.atten = nn.Sequential(
                    nn.Conv1d(dim, self.num_tokens, kernel_size=3, padding=1, bias=False),
                    nn.GELU(),
                    nn.Conv1d(self.num_tokens, self.num_tokens, kernel_size=3, padding=1, bias=False),
                    nn.GELU(),
                    nn.Conv1d(self.num_tokens, self.num_tokens, kernel_size=3, padding=1, bias=False),
                    nn.GELU(),
                    nn.Conv1d(self.num_tokens, self.num_tokens, kernel_size=3, padding=1, bias=False),
                    nn.Sigmoid())

    def forward(self, inputs): # [b, n1, C]

        feat = self.norm(inputs).transpose(1,2) # b, c, n1
        # feat = inputs.transpose(1,2)
        feat = self.atten(feat) 
        b, n2, n1 = feat.size() # n2是新的token个数，n1是第一次的token个数
        feat = feat.reshape((b, n2, n1, 1)) # [b, n2, n1, 1]

        feat_inputs = inputs
        feat_inputs = feat_inputs.unsqueeze(1)  # [b, 1, n1, c] 
        feat_inputs = torch.sum(feat * feat_inputs, axis=2) # b, n2, c

        return feat_inputs



class FSAttention(nn.Module):
    """Factorized Self-Attention"""

    def __init__(self, dim, heads=8, dropout=0., hidden_dim=128):
        super().__init__()
        
        dim_head = dim // heads 
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8) # for 56*56
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4) # for 28*28
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # for 14*14

        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)

        self.norm = nn.LayerNorm(dim)
        # self.norm_ffn = nn.LayerNorm(dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
        
    def forward(self, x, key, value, key1, value1): # x:query
        # x=query: bt, c, h, w
        # key:     bt, num_token, c,    key1:   bt, num_token2, c 
        # value:   bt, num_token, c     value1: bt, num_token2, c
        
        h = self.heads
        b, c, height, w = x.shape

        if height == 56:
            x = self.pool8(x)             # b, c, h//8, w//8
        elif height == 28:
            x = self.pool4(x)             # b, c, h//4, w//4
        elif height == 14:
            x = self.pool2(x)             # b, c, h//2, w//2

        x = x.reshape(b, c, -1).transpose(1,2)
        x = self.norm(x)
        query = self.linear_q(x)          # b, h//4*w//4, c
        q = rearrange(query, 'b n (h d) -> b h n d', h=h) 

        # 第一组的key和value 
        key = self.linear_k(self.norm(key))     # b, num_token, c
        value = self.linear_v(self.norm(value)) # b, num_token, c

        k, v = map(lambda t: rearrange(t, 'b m (h d) -> b h m d', h=h), [key, value])
        dots = einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
        attn = self.attend(dots) 

        out1 = einsum('b h n m, b h m d -> b h n d', attn, v)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out1 = self.to_out(out1)

        # 第二组的key和value
        key1 = self.linear_k(self.norm(key1))     # b, num_token2, c
        value1 = self.linear_v(self.norm(value1)) # b, num_token2, c

        k1, v1 = map(lambda t: rearrange(t, 'b m (h d) -> b h m d', h=h), [key1, value1])
        dots = einsum('b h n d, b h m d -> b h n m', q, k1) * self.scale
        attn = self.attend(dots) 

        out2 = einsum('b h n m, b h m d -> b h n d', attn, v1)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out2 = self.to_out(out2)

        out = out1 + out2
        x = x + out
        out = self.ffn(x) + x # b h//4*w//4 c

        if height == 56:
            out = out.reshape(b, height//8, w//8, c).permute(0, 3, 1, 2) # b, c, h//8, w//8
        elif height == 28:
            out = out.reshape(b, height//4, w//4, c).permute(0, 3, 1, 2) # b, c, h//4, w//4
        elif height == 14:
            out = out.reshape(b, height//2, w//2, c).permute(0, 3, 1, 2) # b, c, h//2, w//2
        else:
            out = out.reshape(b, height, w, c).permute(0, 3, 1, 2) # b, c, h, w
        # print(out)
        return out


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., hidden_dim=128):
        super().__init__()
        
        dim_head = dim // heads 
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),)
            # nn.Sigmoid())
        
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveAvgPool1d(1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, num_frames, token1, token2): # x:query
        # feat: bt, c, h, w
        # token1: b, n1, c
        # token2: b, n2, c
        # num_frames: 一个序列的帧数
       
        h = self.heads
        b, _, c = token1.size()

        # query = self.pool(feat).squeeze(-1).squeeze(-1).reshape(-1, num_frames, c) # b, t, c
        # query = self.norm(query)
        # token1 = torch.max(input=token1, dim=1, keepdim=False)[0].reshape(-1, num_frames, c) # b, t, c
        # token2 = torch.max(input=token2, dim=1, keepdim=False)[0].reshape(-1, num_frames, c)
        token1 = self.pool1(token1.transpose(1, 2)).squeeze(-1).reshape(-1, num_frames, c)
        token2 = self.pool2(token2.transpose(1, 2)).squeeze(-1).reshape(-1, num_frames, c)
        token = token1 + token2
        token = self.norm(token)

        qkv = self.to_qkv(token).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b m (h d) -> b h m d', h=h), qkv)

        dots = einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
        attn = self.attend(dots) 
        

        out = einsum('b h n m, b h m d -> b h n d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out) # b, t, c

        out = out + token
        out = self.ffn(out) + out # b, t, c

        return out 
        

class ShortModule(nn.Module):
    def __init__(self, dim):
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)



        self.model_salinet = nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(dim),
                        nn.ReLU(),
                        nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                        nn.Sigmoid())
        

    def forward(self, ori_x, conv1_feature, res2_feature):
        pass 









if __name__=='__main__':
    # adaptive aggrate token 
    model2D = TokenLearnerModule(num_tokens=10, dim=512)
    inputs = torch.randn([8, 512, 28, 28]) # bt, c, h, w
    out = model2D(inputs)                  # bt, num_token1, c
    # print(out.size())                    # bt, num_token1, c 第一个尺度

    model1D = TokenLearnerModule1D(num_tokens=4, dim=512)
    out2 = model1D(out) # bt, num_token2, c
    # print(out2.size()) 

    model2 = FSAttention(dim=512, heads=8, dropout=0.0, hidden_dim=128) # 512 = 8*64
    out_ = model2(inputs, out, out, out2, out2)
    # print(out_.size())

    temp = TemporalAttention(dim=512, heads=8, dropout=0., hidden_dim=128)
    out_temp = temp(8, out, out2) # b, t, c 


    out_temp = out_temp.unsqueeze(-1).unsqueeze(-1)        # b, t, c, 1, 1 

    # out_ = out_.reshape(b, t, c, h, w)
    # out_ = out_ + out_temp
    out_ = F.interpolate(out_, [28, 28], mode='nearest')   # bt, c, h, w

    print(out.size())

