import torch
import torch.nn as nn


class ResBlock1dTF(nn.Module):
    def __init__(self, dim, dilation=1, kernel_size=3):
        super().__init__()
        self.block_t = nn.Sequential(
            nn.ReflectionPad1d(dilation * (kernel_size // 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, bias=False, dilation=dilation, groups=dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True)
        )
        self.block_f = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True)
        )
        self.shortcut = nn.Conv1d(dim, dim, 1, 1)

    def forward(self, x):
        return self.shortcut(x) + self.block_f(x) + self.block_t(x)


# 创建 ResBlock1dTF 模型
model = ResBlock1dTF(dim=64)

# 创建随机输入张量
batch_size = 16
input_channels = 64
input_length = 100
input_tensor = torch.randn(batch_size, input_channels, input_length)

# 前向传播
output_tensor = model(input_tensor)

# 打印输入和输出的大小
print("输入张量大小:", input_tensor.size())
print("输出张量大小:", output_tensor.size())