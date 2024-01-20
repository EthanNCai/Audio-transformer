import torch
import torch.nn as nn

# 定义 Transformer 模型
transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, num_decoder_layers=12, batch_first=True)

# 创建更大的输入张量
src = torch.rand((64, 20, 512))  # 源序列形状为 (batch_size, sequence_length, feature_dim)
tgt = torch.rand((64, 5, 512))  # 目标序列形状为 (batch_size, sequence_length, feature_dim)

# 使用 Transformer 模型进行前向传播
out = transformer_model(src, tgt)

print(out.shape)  # 输出的形状为 (batch_size, tgt_sequence_length, feature_dim)