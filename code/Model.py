import torch
import torch.nn as nn
import numpy as np


class DownSampling(nn.Module):
    def __init__(self, input_size):
        super(DownSampling, self).__init__()
        in_channels = 1
        out_channels = 1
        dilation = 1
        kernel_size = 3
        stride = 2

        layers = []
        for i in range(5):
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
            bn = nn.BatchNorm1d(out_channels)
            pool = nn.MaxPool1d(kernel_size=2)
            layers.append(conv)
            layers.append(bn)
            layers.append(pool)
            dilation = 2
            in_channels = out_channels
            out_channels *= 2

        self.layers = nn.Sequential(*layers)
        self.audio_net = FCNet(1360, 1360 // 2, 10)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)  # 展平为一维
        x = self.audio_net(x)  # 使用修改后的输入大小
        return x


class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)  # 将输入展平为向量
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = 22050 * 4
input_data1 = torch.randn(1, 1, input_size)
input_data2 = torch.randn(1, 1, input_size)
input_data = torch.cat((input_data1, input_data2), dim=0)
print(input_data.shape)
# 输出：torch.Size([2, 1, 88200])

down_sampling = DownSampling(input_size)
output_data = down_sampling(input_data)
print(output_data.shape)
# 输出：torch.Size([2, 10])