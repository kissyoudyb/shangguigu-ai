__all__ = ["ImageDataset"]

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image
import re

from denoising_config import *

# 定义函数：对图片名按字母数字混合排序
def sorted_alphanum(img_names):
    # 转换函数：将数字部分转为int，将字符串转为小写
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda img_name: [convert(x) for x in re.split(r'([0-9]+)', img_name)]
    return sorted(img_names, key=alphanum_key)

# 自定义数据集类型，元素（noise_img_tensor, original_img_tensor）
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.main_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir))  # 获取目录下所有图片文件名，并按字母数字混合排序

    def __len__(self):
        return len(self.image_names)

    # 传入图片id，获取数据集元素 （x, y）
    def __getitem__(self, idx):
        # 1. 根据索引号，构建图片的完整路径
        image_loc = os.path.join(self.main_dir, self.image_names[idx])
        # 2. 使用 PIL 打开图片
        image = Image.open(image_loc).convert('RGB')
        # 3. 利用transform转换成tensor
        if self.transform is not None:
            tensor_img = self.transform(image)
        else:
            # 如果为None，就抛出异常
            raise ValueError("transform 参数不能为 None！")
        # 4. 向原始图像中增加噪声
        noise_img = tensor_img + NOISE_FACTOR * torch.randn_like(tensor_img)
        noise_img = torch.clamp(noise_img, 0., 1.)

        # 返回 (噪声图片，原始图片)
        return noise_img, tensor_img

if __name__ == "__main__":
    # image_names = os.listdir(IMG_PATH)
    # print(image_names)
    # print(sorted_alphanum(image_names))
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])
    dataset = ImageDataset(IMG_PATH, transform)
    print(len(dataset))