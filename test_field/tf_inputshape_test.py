import torch
import torch.nn as nn


class DownSampling(nn.Module):
    def __init__(self, input_size):
        super(DownSampling, self).__init__()
        dilation = 2
        kernel_size = 2
        stride = 2

        layers = [
            nn.Conv1d(1, 4, kernel_size=5, stride=stride, dilation=dilation),
            nn.MaxPool1d(kernel_size=5, stride=2),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(4, 16, kernel_size=5, stride=stride, dilation=dilation),
            nn.MaxPool1d(kernel_size=5, stride=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, True),

        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


# 创建DownSampling模块实例
input_size = 80021  # 输入的序列长度
down_sampling = DownSampling(input_size)

# 创建输入张量
batch_size = 25
in_channels = 1
seq_len = 125
sequence_length = input_size
input_tensor = torch.randn(batch_size, in_channels, sequence_length)

# 前向传播
output_tensor = down_sampling(input_tensor)

# 输出形状
print("In Shape:", input_tensor.shape)
print("Out Shape:", output_tensor.shape)

# Equal to a Flatten operation.

total_len = output_tensor.numel()//batch_size
feature_size = None
for i in range(256, 0, -1):
    if total_len % i == 0:
        print(total_len, i)
        feature_size = i
        break

viewed_output = output_tensor.view(batch_size, -1, feature_size)

print("viewed Shape:", viewed_output.shape)
