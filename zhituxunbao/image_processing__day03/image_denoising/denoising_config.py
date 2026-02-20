# 数据预处理配置
IMG_PATH = "/root/datasets/imagedata/dataset"
IMG_HEIGHT = 68
IMG_WIDTH = 68

# 随机性与数据集划分
SEED = 42               # 随机数种子
TRAIN_RATIO = 0.75      # 训练集划分比例
TEST_RATIO = 1 - TRAIN_RATIO
NOISE_FACTOR = 0.5      # 噪声因子

# 训练超参数设置
LEARNING_RATE = 0.001   # 学习率
EPOCHS = 50             # 训练总轮次
TRAIN_BATCH_SIZE = 128   # mini-batch大小
TEST_BATCH_SIZE = 128

# 模块名称和保存模型参数的文件名
PROJECT_PACKAGE_NAME = 'image_denoising'
DENOISER_MODEL_NAME = 'denoiser.pt'