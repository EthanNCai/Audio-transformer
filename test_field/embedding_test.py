import torch
import torch.nn as nn

# 创建 Linear 层
input_size = 152
output_size = 512
linear = nn.Linear(input_size, output_size)

batch_size = 32
seq_len = 666
raw_feature_len = 152

input_tensor = torch.randn(batch_size, seq_len, raw_feature_len)
print(input_tensor.shape)

# 将输入张量调整为形状 [32*666, 152]
input_tensor_flat = input_tensor.view(-1, raw_feature_len)

# 前向传播
embedded_tensor_flat = linear(input_tensor_flat)

# 将嵌入张量调整为形状 [32, 666, 512]
embedded_tensor = embedded_tensor_flat.view(batch_size, seq_len, output_size)

print(embedded_tensor.shape)