# SEF_UNet
SEF_UNet-A UNet algorithm for corn leaf spot segmentation
## 一、简介
我们构建了包含 857 幅图像的病叶数据集，并提出了一种高可用性玉米叶斑病分割算法 SEF-UNet。该算法基于 Res-UNet 结构，融合 SElayer 和 ELA 注意力机制，并设计特征融合网络(fusenet)以增强多层特征利用。实验结果表明，SEF-UNet 在 平均联合交集（mIOU）、平均像素准确率（mPA）、平均精度（mPrecision）和平均召回率（mRecall）方面分别达到 92.62%、95.74%、96.63% 和 95.64%。与 U主流方法对比，SEF-UNet 显著提升了病叶图像分割精度，为病害监测和严重程度评估提供了有效技术支持。
## 二、项目目录介绍
- Medical_Datasets ---存储医药的数据集
- R_miou_out       --- 模型训练后运行get_miou.py所得的文件，该文件是为了与miou_out区分重命名的
- VOCdevkit        ---存储要训练的数据集，训练时按照该格式
- Datasets         ---用于训练后做预测的数据
- Img ---同上
- Img_out ---- 预测的结果
- miou_out ----同R_miou_out一样
- Nets ---该文件夹用于网络结构的设计
    Nets/Attention.py  该网络所用到的注意力机制，主要用的是ELA
    Nets/cat.py    该网络的fusenet的实现
    Nets/unet.py   该文件中实现了SELayer，同时调用了cat,ELA,resnet
- Utils  ---模型的工具文件夹
- draw_model_comparison.py --画对比结果的
- get_miou.py   模型训练完后，修改unet.py的参数后使用最好的权重运行该文件获取MIOU
- predict.py --- 模型预测
- Train.py   --模型训练文件
- Unet.py  --实现所有功能的unet，与Nets/unet.py不同
- voc_annotation.py   --数据集划分
## 三、模型训练
### 准备工作
#### 下载所需的包库
`Pip install -r requirements.txt`
#### 下载权重文件
[unet_vgg_voc.pth](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_vgg_voc.pth)  
[unet_resnet_voc.pth](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_resnet_voc.pth)  
下载好后放置于model_data文件夹中(没有新建)  
[best_epoch_weights.pth]( https://pan.baidu.com/s/13icThQWRslez7DdgAybZNw?pwd=1234)
该权重为模型修改后训练玉米病斑数据集得到的  
#### 所需环境
[torch == 1.9.0](https://pan.baidu.com/s/1QH04503MJxWc8KIcopxkzA?pwd=1234)   
[torchvision == 0.10.0](https://pan.baidu.com/s/1AYb5ryFrBDQsvP-ubVes3A?pwd=1234)  
适用于3060显卡，cuda11.1
#### 玉米数据集
[Corn leaf Spot.zip](https://pan.baidu.com/s/1Z3Pe2vTh_EsJo549Rb9aqQ?pwd=1234)  

|![图1.数据集中的原始图像和数据增强后的相应图像](/paper_img/1.jpg)|
|:--:|
| **图1.数据集中的原始图像和数据增强后的相应图像** |
-------
| ![图2.原始图像和标注真实的样本](/paper_img/2.png) |
|:--:|
| **图2.原始图像和标注真实的样本** |


### 训练玉米病斑数据集
1、将我提供的voc数据集放入VOCdevkit/VOC2007中（无需运行voc_annotation.py）。  
2、运行train.py进行训练，默认参数已经对应voc数据集所需要的参数了。  
### 训练自己的数据集
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。  
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4、在训练前利用voc_annotation.py文件生成对应的txt。  
5、unet.py中的num_classes为分类个数+1，model_path选择下载好的权重路径，做预测时可改为自己的  
6、train.py的num_classes为分类个数+1，model_path选择下载好的权重路径  
7、运行train.py即可开始训练。  
### 获取IOU等数据
1、unet.py中的model_path选择训练好的最好权重  
2、get_miou.py中修改name_classes、num_class  
3、运行get_miou.py  
### 预测 
1、unet.py中的model_path选择训练好的最好权重  
2、predict.py中修改name_classes  
3、运行predict.py  
### 分割模型
U-Net 的分割精度受到限制，这主要是由于病变与土壤之间的颜色相似，以及病变通常具有的不规则形状和模糊边界。  
深度是捕捉更精细特征信息并避免假阳性和假阴性等问题的关键因素之一。因此，我们选择了 ResNet50-Unet 基础网络。  
在选择网络深度时，平衡性能和计算资源消耗至关重要。虽然有更深的 ResNet 网络，但考虑到后续轻量级部署的可能性，我们选择了 ResNet50，以在性能和资源利用率之间取得平衡。  

| ![图3.Resnet50-UNet](/paper_img/3.jpg)|
|:--:|
| **图3.Resnet50-UNet** |

### 提出的模型
#### 编码器中的SElayer
为了获得更复杂的目标细节，同时有效过滤掉来自其他通道的无关信息，我们在编码器中引入了注意力网络 SElayer。这意味着它被嵌入到 Resnet50 中。  

|![图4.SElayer的结构图](/paper_img/4.png)|
|:--:|
| **图4.SElayer的结构图** |

SE 层仅置于 ResNet50 网络的瓶颈区（图 5），以最大限度地提高速度和精度。具体来说，SE 模块是在瓶颈单元的 conv3x3 运算之后添加的，缩减系数为 8 (r=8)。  

|![图5.瓶颈设计](/paper_img/5.png)|
|:--:|
| **图5.瓶颈设计** |

#### 层级连接中的ELA注意力
为了尽量减少外来背景噪声对分割结果的影响，我们在级联拼接过程的每一层都加入了 ELA 注意模块

|![图6.ELA结构](/paper_img/6.png)|
|:--:|
| **图6.ELA结构** |

#### 特征融合网络（FuseNet）
该结构采用侧向连接（Lateral Connection）方法将解码器部分的四个特征图输出融合在一起。这种方法有效地保留了不同尺度的信息，提高了模型的信息利用效率。通过融合多个特征图，提高了模型的泛化能力，增加了最终输出的分辨率，降低了过拟合的风险。

|![图7.FuseNet模型结构](/paper_img/7.png)|
|:--:|
| **图7.FuseNet模型结构** |

#### SEF_UNet

|![图8.SEF-UNet结构](/paper_img/8.png)|
|:--:|
| **图8.SEF-UNet结构** |

### 结果
#### SEF_UNet模型的训练

|![图9.SEF-UNet的损失值随迭代次数变化的曲线](/paper_img/9.jpg)|
|:--:|
| **图9.SEF-UNet的损失值随迭代次数变化的曲线** |
<table>
    <caption><strong>表4. 测试集中混淆矩阵分析的统计参数</strong></caption>
    <tr align="center">
        <th>参数</th>
        <th>背景</th>
        <th>玉米叶</th>
        <th>玉米叶斑病</th>
    </tr>
    <tr align="center">
        <td>Pixel count</td>
        <td>284013830</td>
        <td>628134412</td>
        <td>92736054</td>
    </tr>
    <tr align="center">
        <td>TP</td>
        <td>279599386</td>
        <td>620723089</td>
        <td>86634858</td>
    </tr>
    <tr align="center">
        <td>FN</td>
        <td>2433697</td>
        <td>12853207</td>
        <td>10101196</td>
    </tr>
    <tr align="center">
        <td>FP</td>
        <td>10020086</td>
        <td>10420086</td>
        <td>5031296</td>
    </tr>
    <tr align="center">
        <td>TN</td>
        <td>7848141</td>
        <td>23273293</td>
        <td>16732492</td>
    </tr>
    <tr align="center">
        <td>Recall</td>
        <td>0.991</td>
        <td>0.980</td>
        <td>0.896</td>
    </tr>
    <tr align="center">
        <td>Precision</td>
        <td>0.969</td>
        <td>0.983</td>
        <td>0.945</td>
    </tr>
    <tr align="center">
        <td>mPrecision</td>
        <td colspan="3">0.966</td>
    </tr>
</table>

#### 消融实验结果

<table border="1" style="border-collapse: collapse; text-align: center; width: 100%;">
    <caption><strong>表5. 消融实验结果，每个项目的最大数据以粗体显示</strong></caption>
    <tr>
        <th>组</th>
        <th>SElayer</th>
        <th>ELA</th>
        <th>FuseNet</th>
        <th>mIOU</th>
        <th>mPA</th>
        <th>mPrecision</th>
        <th>Recall</th>
    </tr>
    <tr>
        <td>1</td>
        <td>×</td>
        <td>×</td>
        <td>×</td>
        <td>91.51%</td>
        <td>92.31%</td>
        <td>95.57%</td>
        <td>93.31%</td>
    </tr>
    <tr>
        <td>2</td>
        <td>√</td>
        <td>×</td>
        <td>×</td>
        <td>89.46%</td>
        <td>93.13%</td>
        <td>95.50%</td>
        <td>93.13%</td>
    </tr>
    <tr>
        <td>3</td>
        <td>×</td>
        <td>√</td>
        <td>×</td>
        <td>90.95%</td>
        <td>94.11%</td>
        <td>96.22%</td>
        <td>94.11%</td>
    </tr>
    <tr>
        <td>4</td>
        <td>×</td>
        <td>×</td>
        <td>√</td>
        <td>91.76%</td>
        <td>95.30%</td>
        <td>95.86%</td>
        <td>95.32%</td>
    </tr>
    <tr>
        <td>5</td>
        <td>√</td>
        <td>√</td>
        <td>×</td>
        <td>92.48%</td>
        <td>95.61%</td>
        <td>96.10%</td>
        <td>95.61%</td>
    </tr>
    <tr>
        <td>6</td>
        <td>√</td>
        <td>×</td>
        <td>√</td>
        <td>91.25%</td>
        <td>95.06%</td>
        <td>95.53%</td>
        <td>95.06%</td>
    </tr>
    <tr>
        <td>7</td>
        <td>×</td>
        <td>√</td>
        <td>√</td>
        <td>90.41%</td>
        <td>95.01%</td>
        <td>94.63%</td>
        <td>95.01%</td>
    </tr>
    <tr>
        <td><strong>8</strong></td>
        <td><strong>√</strong></td>
        <td><strong>√</strong></td>
        <td><strong>√</strong></td>
        <td><strong>92.62%</strong></td>
        <td><strong>95.74%</strong></td>
        <td><strong>96.63%</strong></td>
        <td><strong>95.64%</strong></td>
    </tr>
</table>

#### 与原始模型的比较

|![图10.训练数据的mIOU箱线图](/paper_img/10.png)|
|:--:|
| **图10.训练数据的mIOU箱线图** |
---------
|![图11. 不同模型下各种结果的折线图](/paper_img/11.png)|
|:--:|
| **图11. 不同模型下各种结果的折线图** |
---------
|![图12. 与原始模型分割结果的对比](/paper_img/12.jpg)|
|:--:|
| **图12. 与原始模型分割结果的对比** |


#### 与其他分割算法的实验结果对比

|![图13.不同模型的mIOU比较](/paper_img/13.jpg)|
|:--:|
| **图13.不同模型的mIOU比较** |
--------
|![图14.与其他分割模型的对比](/paper_img/14.jpg)|
|:--:|
| **图14.与其他分割模型的对比** |

### 总结
本研究构建了一个大规模的玉米叶片病害数据集，为作物病害领域的视觉研究提供了重要的数据支持。我们提出了一种针对玉米叶片叶斑病的高可用性分割算法，称为 SEF-UNet，它使用 Res-UNet 作为骨干网络。该算法参考了 SElayer 和 ELA（Efficient Local Attention，高效局部注意力）。同时，我们还实现了一个以各层输出为重点的特征融合网络。在训练阶段，我们采用骰子损失（Dice Loss）和交叉熵损失（CE Loss）的组合作为损失函数。
实验结果表明，与原始网络相比，本文提出的方法在mIOU、mPA、mPrecision和mRecall等指标上分别提高了1.11%、3.41%、1.06%和2.33%。同时，与主流的语义分割算法相比，SEF-UNet 的性能都有很大提高。这项研究为农业领域的智能病害监测提供了有效的工具和方法。
本研究提出了一种高精度玉米病叶分割算法。它可以为相关研究人员提供一种准确、快速计算玉米叶片面积和病斑面积的方法。它还可以帮助农民在农业生产中准确识别玉米叶斑病，初步帮助农民评估病害的严重程度，实现精准防控，减少病害蔓延造成的损失。
