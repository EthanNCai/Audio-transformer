import torch
import torch.nn as nn

# 定义输入张量的大小和卷积核的参数
batch_size = 32
in_channels = 1
input_len = 800
kernel_size = 3
dilation = 2


input_tensor = torch.randn(batch_size, in_channels, input_len)

stream_line = nn.Sequential(
    nn.ReflectionPad1d(dilation * (kernel_size // 2)),
    nn.Conv1d(1, 3, kernel_size, dilation=dilation),
    nn.ReflectionPad1d(dilation * (kernel_size // 2)),
    nn.Conv1d(3, 9, kernel_size, dilation=dilation),
)

print("Input tensor shape:", input_tensor.shape)
output_tensor = stream_line(input_tensor)
print("Output tensor shape:", output_tensor.shape)
