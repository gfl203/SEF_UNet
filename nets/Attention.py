import torch
import torch.nn as nn
class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu(x + 3.0) / 6.0

# 创建 HSwish 激活函数实例

#注意力机制
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
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




class EfficientLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(EfficientLocalizationAttention, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        # print(x_h.shape, x_w.shape)
        # 在两个维度上应用注意力
        return x * x_h * x_w


# 示例用法 ELABase(ELA-B)
# if __name__ == "__main__":
#     # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
#     dummy_input = torch.randn(2, 64, 32, 32)
#
#     # 初始化模块
#     ela = EfficientLocalizationAttention(channel=dummy_input.size(1))
#
#     # 前向传播
#     output = ela(dummy_input)
#     # 打印出输出张量的形状，它将与输入形状相匹配。
#     print(f"输出形状: {output.shape}")

"""
为了在考虑参数数量的同时优化ELA的性能，作者引入了四种方案: ELA-Tiny(ELA-T)，ELABase(ELA-B)，ELA-Smal(ELA-S)和ELA-Large(ELA-L)。
1.ELA-T的参数配置定义为 kernel size=5,groups =in channels， num group=32:
2.ELA-B的参数配置定义为 kernel size=7，groups =in_channels， num_group =16:
3.ELA-S的参数配置为 kernel size=5,groups=in_channels/8, num_group=16。
4.ELA-L的参数配置为 kernel_size=7,groups=in _channels /8，num_group=16 。
"""




