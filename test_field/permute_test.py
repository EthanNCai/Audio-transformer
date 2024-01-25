import torch

# 创建一个示例张量
x = torch.randn(2, 3, 4)

# 使用permute和contiguous进行维度变换
x_permuted = x.permute(0, 2, 1).contiguous()

# 打印原始张量和变换后的张量
print("原始张量：\n", x)
print("变换后的张量：\n", x_permuted)

# 验证维度变换是否正确
print("维度是否变换正确：", x_permuted.shape == (2, 4, 3))