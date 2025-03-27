import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from nets.resnet import resnet50
import torch.nn.functional as F
# 注意力机制
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


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=3, pretrained=False):
        super(Unet, self).__init__()

        self.intermediate_features = []
        self.resnet = resnet50(pretrained=pretrained)
        in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]

        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])  # 修改这里
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])  # 修改这里
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])  # 修改这里
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=256)
        self.cbam3 = CBAM(channel=512)
        self.cbam4 = CBAM(channel=1024)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(self.cbam4(feat4)+feat4, feat5)
        up3 = self.up_concat3(self.cbam3(feat3)+feat3, up4)
        up2 = self.up_concat2(self.cbam2(feat2)+feat2, up3)
        up1 = self.up_concat1(self.cbam1(feat1)+feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)
        self.features = up1
        x = self.final(up1)

        return x


import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 加载你的 UNet 模型
# 假设你的模型名为 UNet，并且已经训练好了
model = Unet()
model.load_state_dict(torch.load('RES_logs/best_epoch_weights.pth'))
model.eval()

# 定义输入图像的预处理步骤
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 根据你的模型输入尺寸调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 的均值和标准差
])

# 读取输入图像
image = Image.open('img/1.jpg')
image_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度

# 在模型上运行输入图像
with torch.no_grad():
    output = model(image_tensor)
print(output)

# 获取热力图
# 假设 output 是模型的最后一层输出，即未经过 softmax 或 sigmoid 的原始分数图
heatmap = output[0, 0, :, :].squeeze()

# 对热力图进行归一化
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255

# 将热力图转换为 PIL 图像
heatmap = heatmap.astype(np.uint8)
heatmap_img = Image.fromarray(heatmap)
# 显示热力图
plt.imshow(heatmap_img, cmap='gray')
plt.show()
