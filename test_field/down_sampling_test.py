import torch
import torch.nn as nn
import torch.nn.functional as F


class AADownsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(AADownsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        ha = torch.arange(1, filt_size // 2 + 1 + 1, 1)
        a = torch.cat((ha, ha.flip(dims=[-1, ])[1:])).float()
        a = a / a.sum()
        filt = a[None, :]
        self.register_buffer('filt', filt[None, :, :].repeat((self.channels, 1, 1)))

    def forward(self, x):
        x_pad = F.pad(x, (self.filt_size // 2, self.filt_size // 2), "reflect")
        y = F.conv1d(x_pad, self.filt, stride=self.stride, padding=0, groups=x.shape[1])
        return y


class MidLayer(nn.Module):
    def __init__(self, input_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size // 2, output_size)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x_):
        x_ = self.linear1(x_)
        x_ = self.relu(x_)
        x_ = self.linear2(x_)
        x_ = self.relu(x_)
        return x_


class Down(nn.Module):
    def __init__(self, channels, stride=2, k=3):
        super().__init__()
        kk = stride + 1
        self.down = nn.Sequential(
            nn.ReflectionPad1d(kk // 2),
            nn.Conv1d(channels, channels * 2, kernel_size=kk, stride=1, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.LeakyReLU(0.2, True),
            AADownsample(channels=channels * 2, stride=stride, filt_size=k)
        )

    def forward(self, x):
        x = self.down(x)
        return x


# 创建一个 Down 模块的实例
channels = 16
stride = 2
k = 3
down_module = Down(channels, stride, k)
down_module1 = Down(channels * 2, stride, k)
down_module2 = Down(channels * 4, stride, k)
down_module3 = Down(channels * 8, stride, k)
down_module4 = Down(channels * 16, stride, k)

# 随机生成一个输入张量
batch_size = 4
input_channels = channels
input_length = 8000
x = torch.randn(batch_size, input_channels, input_length)

# 打印输入张量的形状
print("输入张量形状: ", x.shape)

# 输入 Down 模块前的张量形状
output_before = down_module.down[0](x)
print("Down模块前的输出张量形状: ", output_before.shape)

# 输入 Down 模块后的张量形状
output_after = down_module(x)
output_after = down_module1(output_after)
output_after = down_module2(output_after)
output_after = down_module3(output_after)
output_after = down_module4(output_after)
print("Down模块后的输出张量形状: ", output_after.shape)

# [batch_size, channels, feature_size(variable)]
