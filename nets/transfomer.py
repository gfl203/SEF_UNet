from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from self_attention_cv import *
from mmcv.cnn import build_norm_layer


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()  # 初始化父类
        self.downsample = nn.Sequential(  # 下采样层，用于降低特征图的维度
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        width = int(out_channels * (base_width / 64))  # 计算中间通道数
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)  # 第一个卷积层
        self.norm1 = nn.BatchNorm2d(width)  # 第一个批量归一化层
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1,
                               bias=False)  # 第二个卷积层
        self.norm2 = nn.BatchNorm2d(width)  # 第二个批量归一化层
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)  # 第三个卷积层
        self.norm3 = nn.BatchNorm2d(out_channels)  # 第三个批量归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

    def forward(self, x):
        x_down = self.downsample(x)  # 下采样操作
        x = self.conv1(x)  # 第一个卷积操作
        x = self.norm1(x)  # 第一个批量归一化
        x = self.relu(x)  # ReLU激活
        x = self.conv2(x)  # 第二个卷积操作
        x = self.norm2(x)  # 第二个批量归一化
        x = self.relu(x)  # ReLU激活
        x = self.conv3(x)  # 第三个卷积操作
        x = self.norm3(x)  # 第三个批量归一化
        x = x + x_down  # 残差连接
        x = self.relu(x)  # ReLU激活
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()  # 调用父类构造函数

        self.head_num = head_num  # 多头的数量
        self.dk = (embedding_dim // head_num) ** (1 / 2)  # 缩放因子，用于缩放点积注意力

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)  # 线性层，用于生成查询（Q）、键（K）和值（V）
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)  # 输出线性层

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)  # 通过线性层生成Q、K、V

        query, key, value = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.head_num))  # 将Q、K、V重塑为多头注意力的格式
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk  # 计算点积注意力的能量

        if mask is not None:  # 如果提供了掩码，则在能量上应用掩码
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)  # 应用softmax函数，得到注意力权重

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)  # 应用注意力权重到值上

        x = rearrange(x, "b h t d -> b t (h d)")  # 重塑x以准备输出
        x = self.out_attention(x)  # 通过输出线性层

        return x


# 定义MLP模块
class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()  # 调用父类构造函数

        self.mlp_layers = nn.Sequential(  # 定义MLP的层
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),  # GELU激活函数
            nn.Dropout(0.1),  # Dropout层，用于正则化
            nn.Linear(mlp_dim, embedding_dim),  # 线性层
            nn.Dropout(0.1)  # Dropout层
        )

    def forward(self, x):
        x = self.mlp_layers(x)  # 通过MLP层
        return x


# 定义Transformer编码器块
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()  # 调用父类构造函数

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)  # 多头注意力模块
        self.mlp = MLP(embedding_dim, mlp_dim)  # MLP模块

        self.layer_norm1 = nn.LayerNorm(embedding_dim)  # 第一层归一化
        self.layer_norm2 = nn.LayerNorm(embedding_dim)  # 第二层归一化

        self.dropout = nn.Dropout(0.1)  # Dropout层

    def forward(self, x):
        _x = self.multi_head_attention(x)  # 通过多头注意力模块
        _x = self.dropout(_x)  # 应用dropout
        x = x + _x  # 残差连接
        x = self.layer_norm1(x)  # 第一层归一化

        _x = self.mlp(x)  # 通过MLP模块
        x = x + _x  # 残差连接
        x = self.layer_norm2(x)  # 第二层归一化

        return x


# 定义ViT模型
class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim, block_num, patch_dim,
                 classification=True, num_classes=1):
        super().__init__()  # 调用父类构造函数

        self.patch_dim = patch_dim  # 定义patch的维度
        self.classification = classification  # 是否进行分类
        self.num_tokens = (img_dim // patch_dim) ** 2  # 计算tokens的数量
        self.token_dim = in_channels * (patch_dim ** 2)  # 计算每个token的维度

        self.projection = nn.Linear(self.token_dim, embedding_dim)  # 线性层，用于将patches投影到embedding空间
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))  # 可学习的embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))  # 类别token

        self.dropout = nn.Dropout(0.1)  # Dropout层

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)  # Transformer编码器

        if self.classification:  # 如果是分类任务
            self.mlp_head = nn.Linear(embedding_dim, num_classes)  # 分类头

    def forward(self, x):
        img_patches = rearrange(x,  # 将输入图像重塑为patches序列
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape  # 获取批次大小、tokens数量和通道数

        project = self.projection(img_patches)  # 将patches投影到embedding空间
        # token = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=batch_size)  # 重复cls_token以匹配批次大小
        #repeat 函数在 PyTorch 中的用法是只有两个参数的，也就是重复一个张量。为了扩展一个张量以匹配批次大小，您可以使用 expand 或者 tile 函数。下面是修正后的代码：
        token = self.cls_token.expand(batch_size, -1, -1)  # 扩展cls_token以匹配批次大小

        patches = torch.cat((token, project), dim=1)  # 将cls_token和投影后的patches拼接
        patches += self.embedding[:tokens + 1, :]  # 将可学习的embedding添加到patches

        x = self.dropout(patches)  # 应用dropout
        x = self.transformer(x)  # 通过Transformer编码器
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]  # 如果是分类任务，使用cls_token的输出；否则，使用patches的输出

        return x


# 定义解码器中的瓶颈层
class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()  # 初始化父类
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)  # 上采样层
        self.layer = nn.Sequential(  # 解码器层
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)  # 上采样操作
        if x_concat is not None:  # 如果有额外的特征图进行拼接
            x = torch.cat([x_concat, x], dim=1)  # 在通道维度上拼接
        x = self.layer(x)  # 通过解码器层
        return x


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, img_dim=512, in_channels=3, out_channels=64, head_num=8, mlp_dim=152, block_num=2, patch_dim=16):
        super().__init__()  # 初始化父类
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)  # 第一个卷积层
        self.norm1 = nn.BatchNorm2d(64)  # 第一个批量归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 4, stride=2)  # 第一个编码器瓶颈层
        self.encoder2 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)  # 第二个编码器瓶颈层
        self.encoder3 = EncoderBottleneck(out_channels * 8, out_channels * 16, stride=2)  # 第三个编码器瓶颈层
        self.vit_img_dim = img_dim // patch_dim  # ViT的图像维度
        self.vit = ViT(self.vit_img_dim, out_channels * 16, out_channels * 16,  # ViT模型
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)
        self.conv2 = nn.Conv2d(out_channels * 16, 2048, kernel_size=3, stride=1, padding=1)  # 第四个卷积层
        self.norm2 = nn.BatchNorm2d(2048)  # 第四个批量归一化层

    def forward(self, x):
        x = self.conv1(x)  # 第一个卷积操作
        x = self.norm1(x)  # 第一个批量归一化
        x1 = self.relu(x)  # ReLU激活
        x2 = self.encoder1(x1)  # 第一个编码器瓶颈层
        x3 = self.encoder2(x2)  # 第二个编码器瓶颈层
        x4 = self.encoder3(x3)  # 第三个编码器瓶颈层
        x = self.vit(x4)  # 通过ViT模型
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # 重塑特征图
        x = self.conv2(x)  # 第四个卷积操作
        x = self.norm2(x)  # 第四个批量归一化
        x = self.relu(x)  # ReLU激活
        print("-----------------")
        print("-----------------")
        return [x1, x2, x3,x4, x]