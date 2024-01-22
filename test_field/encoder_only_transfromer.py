import torch
import torch.nn as nn

# 定义 Transformer 模型
embed_dim = 512
n_head = 8
dim_feedforward = 2048
drop_rate = 0.1
n_layers = 8

enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, activation="gelu",
                                       dim_feedforward=dim_feedforward, dropout=drop_rate)
transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim))

# 创建输入张量
src = torch.rand((32, 10, 512))  # 源序列形状为 (batch_size, sequence_length, feature_dim)

# 使用 Transformer 编码器进行特征提取和上下文建模
encoded_src = transformer_enc(src)

# 添加全连接层
fc_layer = nn.Linear(embed_dim, 10)  # 全连接层将编码后的特征维度从 embed_dim 转换为 256
output = fc_layer(encoded_src)

print(output.shape)  # 输出的形状为 (batch_size, sequence_length, 256)sequence_length, feature_dim)
