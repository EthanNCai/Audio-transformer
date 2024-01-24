import torch
import torch.nn.functional as F

# 输入张量的形状为 [32, 1, 7777]
input_tensor = torch.randn(32, 1, 7777)

# 目标形状 [32, ?, 512]
target_shape = (32, -1, 512)

# 计算需要补零的数量
num_padding = target_shape[2] - input_tensor.size(2)

# 在最后一维进行填充，补零数量为 num_padding
padded_tensor = F.pad(input_tensor, (0, num_padding))

# 调整张量形状为目标形状
output_tensor = padded_tensor.view(*target_shape)

print(output_tensor.shape)  # 输出: torch.Size([32, ?, 512])