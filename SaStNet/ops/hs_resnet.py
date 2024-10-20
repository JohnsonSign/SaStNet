import torch
import torch.nn as nn
# HSBlock写的还是又bug，只不过是可能不会遇见那些特殊的情况
class HSBlock(nn.Module):
    '''
    替代3x3卷积
    '''
    def __init__(self, in_ch, s=8):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        :param w: 滤波器宽度（卷积核的输出通道）
        '''
        super(HSBlock, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        # 避免无法整除通道数,当通道无法整除时,划分方式就是in_ch//s+1,in_ch%s+1,可以自己举个例子试试
        in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s + 1)
        self.module_list.append(nn.Sequential()) ## 这里加上这一个空的，后面也没有用，可能是为了逻辑上确实第一个split啥都没做，方便理解
        acc_channels = 0
        for i in range(1,self.s):
            if i == 1:
                channels=in_ch
                acc_channels=channels//2
            elif i == s - 1:
                channels = in_ch_last + acc_channels
            else:
                channels=in_ch+acc_channels
                acc_channels=channels//2
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels)) 
        #self.initialize_weights() 因为之前已经初始化过了
        # print(self.module_list)

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1)) # 按照通道维度分为s块，最后一个可能维度少
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class HSBottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, s=8):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        #################################################################
        # 特征图尺度改变，用原生ResBlock
        if stride != 1:
            self.conv_3x3 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            out_ch = out_channels
        # 特征图尺度不变，用HSBlock
        else:
            self.conv_3x3 = HSBlock(in_ch=out_channels, s=s)
        #################################################################
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * HSBottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * HSBottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * HSBottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * HSBottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * HSBottleNeck.expansion)
            )

    def forward(self, x):
        x_hs = self.conv_1x1(x)
        x_hs = self.conv_3x3(x_hs)
        x_hs = self.conv_1x1_2(x_hs)
        return nn.ReLU(inplace=True)(x_hs + self.shortcut(x))


class HS_ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=512):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x) # x input size: [64,3,224,224]
        output = self.maxpool(output)
        #print('after conv1-bn-relu-maxpool size: ',output.size()) # [64,64,56,56]
        output = self.conv2_x(output)
        #print('after conv2_x size: ',output.size()) # [64, 256, 224, 224]
        output = self.conv3_x(output)
        #print('after conv3_x size: ',output.size()) # [64, 512, 112, 112]
        output = self.conv4_x(output)
        #print('after conv4_x size: ',output.size()) # [64, 1024, 56, 56]
        output = self.conv5_x(output)
        #print('after conv5_x size: ',output.size()) # [64, 2048, 28, 28]
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def hs_resnet50():
    """ return a ResNet 50 object
    """
    # return HS_ResNet(BottleNeck, [3, 4, 6, 3])
    return HS_ResNet(HSBottleNeck, [3, 4, 6, 3])


def hs_resnet101():
    """ return a ResNet 101 object
    """
    # return HS_ResNet(BottleNeck, [3, 4, 23, 3])
    return HS_ResNet(HSBottleNeck, [3, 4, 23, 3])


def hs_resnet152():
    """ return a ResNet 152 object
    """
    # return HS_ResNet(BottleNeck, [3, 8, 36, 3])
    return HS_ResNet(HSBottleNeck, [3, 8, 36, 3])


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0")
    # [batch,channel,H,W]
    feature = torch.rand(1, 3, 224, 224).to(device)
    model = hs_resnet50().to(device).train()
    result = model(feature)
    print(result.size())
    # input = torch.randn((8,16,64,64))
    # model = HSBlock(16)
    # output = model(input)
    # print(output.size())