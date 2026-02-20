import numpy as np
import torch
import random
import os

# 定义函数：对所有模块使用统一的随机数种子
def seed_everything(seed):
    random.seed(seed)   # Python内置随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)    # Python哈希种子
    np.random.seed(seed)
    torch.manual_seed(seed)     # CPU
    torch.cuda.manual_seed(seed)    # GPU
    torch.backends.cudnn.deterministic = True   # 确保CuDNN的确定性
    torch.backends.cudnn.benchmark = False   # 关闭CuDNN性能优化

torch.nn.ConvTranspose2d
