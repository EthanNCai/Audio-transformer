import torch
import torch.nn as nn
import numpy as np


class DownSampling(nn.Module):
    def __init__(self, input_size):
        super(DownSampling, self).__init__()
        dilation = 2
        kernel_size = 2
        stride = 2

        layers = [
            nn.Conv1d(1, 2, kernel_size=5, stride=stride, dilation=dilation),
            nn.MaxPool1d(kernel_size=5, stride=2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(2, 4, kernel_size=5, stride=stride, dilation=dilation),
            nn.MaxPool1d(kernel_size=5, stride=2),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(4, 8, kernel_size=5, stride=stride, dilation=dilation),
            nn.MaxPool1d(kernel_size=5, stride=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(8, 16, kernel_size=5, stride=stride, dilation=dilation),
            nn.MaxPool1d(kernel_size=5, stride=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, True)
        ]

        self.layers = nn.Sequential(*layers)
        self.audio_net = FCNet(5440, 5440 // 2, 10)

    def forward(self, x):

        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)

        x = self.audio_net(x)
        return x


class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()  # 添加额外的ReLU层
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)  # 将输入展平为向量
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu2(out)  # 使用额外的ReLU层
        out = self.fc4(out)
        return out

