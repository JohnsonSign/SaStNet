import torch
from torch import nn
from torch.nn import functional as f


# class EfficientAttention(nn.Module):
    
#     def __init__(self, in_channels, key_channels, head_count, value_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.key_channels = key_channels
#         self.head_count = head_count
#         self.value_channels = value_channels

#         self.keys = nn.Conv2d(in_channels, key_channels, 1)
#         self.queries = nn.Conv2d(in_channels, key_channels, 1)
#         self.values = nn.Conv2d(in_channels, value_channels, 1)
#         self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

#     def forward(self, input_):
#         n, _, h, w = input_.size()
#         keys = self.keys(input_).reshape((n, self.key_channels, h * w))
#         queries = self.queries(input_).reshape(n, self.key_channels, h * w)
#         values = self.values(input_).reshape((n, self.value_channels, h * w))
#         head_key_channels = self.key_channels // self.head_count
#         head_value_channels = self.value_channels // self.head_count
        
#         attended_values = []
#         for i in range(self.head_count):
#             key = f.softmax(keys[
#                 :,
#                 i * head_key_channels: (i + 1) * head_key_channels,
#                 :
#             ], dim=2) ##沿着空间h维度进行softmax
#             query = f.softmax(queries[
#                 :,
#                 i * head_key_channels: (i + 1) * head_key_channels,
#                 :
#             ], dim=1)#沿着通道c维度进行softmax
#             value = values[
#                 :,
#                 i * head_value_channels: (i + 1) * head_value_channels,
#                 :
#             ]
#             context = key @ value.transpose(1, 2) ##??
#             attended_value = (
#                 context.transpose(1, 2) @ query
#             ).reshape(n, head_value_channels, h, w) ##??
#             attended_values.append(attended_value)

#         aggregated_values = torch.cat(attended_values, dim=1)
#         reprojected_value = self.reprojection(aggregated_values)
#         attention = reprojected_value + input_

#         return attention

# if __name__=='__main__':
#     x = torch.randn((8,1,224,224)) ## 
#     model = EfficientAttention(1,2,2,2) # 第一个1表示输入通道，剩下的几个参数可以随机设置
#     print(model(x).size())

class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[:,i * head_key_channels: (i + 1) * head_key_channels,:], dim=2) ##沿着空间h维度进行softmax
            query = f.softmax(queries[:,i * head_key_channels: (i + 1) * head_key_channels,:], dim=1)#沿着通道c维度进行softmax
            value = values[:,i * head_value_channels: (i + 1) * head_value_channels,:]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

if __name__=='__main__':
    x = torch.randn((8,6,224,224)) ## in_channels, key_channels, head_count, value_channels
    model = EfficientAttention(6,10,1,10)## 可以在Conv1d中指定group，这样参数就会少了
    print(model(x).size())
# 第一个1表示输入通道，其参数必须与即将喂入的input的shape是一致的;key和value的channel没事最好设置为一致;head_count最好被设置的channel数整除