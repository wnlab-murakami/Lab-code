import torch
import torch.nn as nn
from complexNN.nn import cConv2d, cLinear, cRelu, cBatchNorm2d, cGelu, cSigmoid

class cSE_block(nn.Module):
    """Complex Squeeze-and-Excitation block."""
    def __init__(self, channel, reduction=16):
        super(cSE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = cLinear(channel, channel // reduction)
        self.fc2 = cLinear(channel // reduction, channel)
        self.relu = cRelu()
        self.sigmoid = cSigmoid()

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channel)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channel, 1, 1)
        return x * y.expand_as(x)
    
class cMBConv2d(nn.Module):
    """Complex Mobile Inverted Residual Bottleneck."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion=6, use_se=False):
        super(cMBConv2d, self).__init__()
        self.use_se = use_se
        mid_channels = in_channels * expansion
        
        self.conv1 = cConv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = cBatchNorm2d(mid_channels)
        self.conv2 = cConv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=mid_channels)
        self.bn2 = cBatchNorm2d(mid_channels)
        self.conv3 = cConv2d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = cBatchNorm2d(out_channels)

        if self.use_se:
            self.se = cSE_block(mid_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = cGelu()(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if identity.shape == out.shape:
            return identity + out
        return out

class cChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super(cChannelShuffle, self).__init__()
        if groups <= 1:
            raise ValueError('groups must be greater than 1.')
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.size()
        
        # グループ数でチャネル数が割り切れることを確認
        if num_channels % self.groups != 0:
            raise ValueError('num_channels must be divisible by groups.')
            
        channels_per_group = num_channels // self.groups

        # 1. Reshape
        # (N, C, H, W) -> (N, g, c/g, H, W)
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # 2. Transpose
        # (N, g, c/g, H, W) -> (N, c/g, g, H, W)
        x = torch.transpose(x, 1, 2).contiguous()

        # 3. Flatten
        # (N, c/g, g, H, W) -> (N, C, H, W)
        x = x.view(batch_size, -1, height, width)

        return x

class cShuffledGroupedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int, stride: int = 1, padding: int = 0):
        super(cShuffledGroupedConv, self).__init__()

        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups.')
            
        self.conv = cConv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups,
            bias=False 
        )
        self.bn = cBatchNorm2d(out_channels)
        self.relu = cRelu(inplace=True)
        self.channel_shuffle = cChannelShuffle(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.channel_shuffle(x)
        return x

class cConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, groups=1, expansion=4):
        super(cConvNeXtBlock, self).__init__()
        self.conv1 = cConv2d(in_channels, out_channels // expansion, kernel_size=1)
        self.bn1 = cBatchNorm2d(out_channels // expansion)
        self.conv2 = cShuffledGroupedConv(out_channels // expansion, out_channels // expansion, kernel_size=kernel_size, groups=groups, stride=stride, padding=padding)
        self.bn2 = cBatchNorm2d(out_channels // expansion)
        self.conv3 = cConv2d(out_channels // expansion, out_channels, kernel_size=1)
        self.bn3 = cBatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = cGelu()(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return identity + out if identity.shape == out.shape else out   