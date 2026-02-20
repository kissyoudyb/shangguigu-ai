import torch
import torch.nn as nn

# 定义神经网络结构类
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # 编码器
        # 三个卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        # 通用池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 解码器
        # 三个转置卷积层
        self.t_conv1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        # 普通卷积层
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # 编码
        x = torch.relu(self.conv1(x))
        # print("Conv1 output shape: ", x.shape)
        x = self.pool(x)
        # print("Pool1 output shape: ", x.shape)
        x = torch.relu(self.conv2(x))
        # print("Conv2 output shape: ", x.shape)
        x = self.pool(x)
        # print("Pool2 output shape: ", x.shape)
        x = torch.relu(self.conv3(x))
        # print("Conv3 output shape: ", x.shape)
        x = self.pool(x)
        # print("Pool3 output shape: ", x.shape)
        # 解码
        x = torch.relu(self.t_conv1(x))
        # print("T_Conv1 output shape: ", x.shape)
        x = torch.relu(self.t_conv2(x))
        # print("T_Conv2 output shape: ", x.shape)
        x = torch.relu(self.t_conv3(x))
        # print("T_Conv3 output shape: ", x.shape)
        x = torch.sigmoid(self.conv_out(x))
        # print("Conv output shape: ", x.shape)
        return x

if __name__ == '__main__':
    input = torch.randn(5, 3, 68, 68)
    model = ConvDenoiser()
    output = model(input)
    print(output.shape)