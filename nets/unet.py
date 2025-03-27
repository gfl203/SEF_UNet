import torch
import torch.nn as nn
from nets.SCConv import ScConv
from nets.resnet import resnet50
from nets.vgg import VGG16
from nets.Attention import *
from nets.cat import *
# class DepthWiseConv(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         # 这一行千万不要忘记
#         super(DepthWiseConv, self).__init__()
#
#         # 逐通道卷积
#         self.depth_conv = nn.Conv2d(in_channels=in_channel,
#                                     out_channels=in_channel,
#                                     kernel_size=3,
#                                     stride=1,
#                                     padding=1,
#                                     groups=in_channel)
#         # groups是一个数，当groups=in_channel时,表示做逐通道卷积
#
#         # 逐点卷积
#         self.point_conv = nn.Conv2d(in_channels=in_channel,
#                                     out_channels=out_channel,
#                                     kernel_size=1,
#                                     stride=1,
#                                     padding=0,
#                                     groups=1)
#
#
#     def forward(self, input):
#         out = self.depth_conv(input)
#         out = self.point_conv(out)
#         return out

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        # self.conv1 = DepthWiseConv(in_size, out_size)
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        # self.conv2 = ScConv(out_size)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # self.ScConv4 = ScConv(out_filters[3])  # 添加ScConv模块

        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # self.ScConv3 = ScConv(out_filters[2])  # 添加ScConv模块
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # self.ScConv2 = ScConv(out_filters[1])

        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        # self.ScConv1 = ScConv(out_filters[0])
        # self.CBR=CBR()
        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                # ScConv(out_filters[0]),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None
        self.cat=FeatureFusionNet()
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone
        self.ELA1 = EfficientLocalizationAttention(channel=64)
        self.ELA2 = EfficientLocalizationAttention(channel=256)
        self.ELA3 = EfficientLocalizationAttention(channel=512)
        self.ELA4 = EfficientLocalizationAttention(channel=1024)
        # # 加入cbam注意力机制
        # self.cbam1 = CBAM(channel=64)
        # self.cbam2 = CBAM(channel=256)
        # self.cbam3 = CBAM(channel=512)
        # self.cbam4 = CBAM(channel=1024)
        # 加入注意力机制
        # self.cbam1 = CBAM(channel=64)
        # self.cbam2 = CBAM(channel=128)
        # self.cbam3 = CBAM(channel=256)
        # self.cbam4 = CBAM(channel=512)
    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        # up4 = self.up_concat4(feat4, feat5)
        # up4 = self.ScConv4(up4)
        # up3 = self.up_concat3(feat3, up4)
        # up3 = self.ScConv3(up3)
        # up2 = self.up_concat2(feat2, up3)
        # up2 = self.ScConv2(up2)
        # up1 = self.up_concat1(feat1, up2)
        # up1 = self.ScConv1(up1)

        # up4 = self.up_concat4(feat4, feat5)
        # up3 = self.up_concat3(feat3, up4)
        # up2 = self.up_concat2(feat2, up3)
        # up1 = self.up_concat1(feat1, up2)


        up4 = self.up_concat4(self.ELA4(feat4)+feat4, feat5)
        # up4 = self.ScConv4(up4)
        up3 = self.up_concat3(self.ELA3(feat3)+feat3, up4)
        # up3 = self.ScConv3(up3)
        up2 = self.up_concat2(self.ELA2(feat2)+feat2, up3)
        # up2 = self.ScConv2(up2)
        up1 = self.up_concat1(self.ELA1(feat1)+feat1, up2)   #
        # up1 = self.ScConv1(up1)

        # up4 = self.up_concat4(self.cbam4(feat4)+feat4, feat5)
        # up3 = self.up_concat3(self.cbam3(feat3)+feat3, up4)
        # up2 = self.up_concat2(self.cbam2(feat2)+feat2, up3)
        # up1 = self.up_concat1(self.cbam1(feat1)+feat1, up2)
        # print(up1.shape)
        # print(up2.shape)
        # print(up3.shape)
        # print(up4.shape)
        up1=self.cat(up1, up2, up3, up4)
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True

# x = torch.randn(1, 16, 16, 16)
net = Unet(backbone='resnet50')

'''方法1，自定义函数 参考自 https://blog.csdn.net/qq_33757398/article/details/109210240'''
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

model_structure(net)