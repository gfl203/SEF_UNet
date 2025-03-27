import os
import cv2
import numpy as np


def merge_mask_with_image(original_image_path, mask_image_path, output_folder):
    # 读取原始图片和mask图片
    original_image = cv2.imread(original_image_path)
    mask_image = cv2.imread(mask_image_path)

    # 确保mask图片的大小与原始图片一致
    mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

    # # 将mask图片的红色通道映射到绿色，绿色通道映射到红色
    # mask_image[:, :, [0, 1]] = mask_image[:, :, [1, 0]]

    # 创建一个新的文件夹，用于保存输出图片
    os.makedirs(output_folder, exist_ok=True)

    # 生成不同透明度的输出图片
    for alpha in range(45, 60, 1):
        merged_image = cv2.addWeighted(original_image, 1 - alpha / 100, mask_image, alpha / 100, 0)
        output_image_name = f"{os.path.splitext(os.path.basename(original_image_path))[0]}_{alpha}.png"
        output_image_path = os.path.join(output_folder, output_image_name)
        cv2.imwrite(output_image_path, merged_image)


# 调用函数并传入文件路径和输出文件夹
original_image_path = r'datasets/JPEGImages/3.jpg'
mask_image_path = r'datasets/SegmentationClass/3.png'
output_folder = r'D:\output_images'  # 新的文件夹路径
merge_mask_with_image(original_image_path, mask_image_path, output_folder)
