import torch
import torch.nn as nn

class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=64, n_layers=6, nhead=6, n_classes=None, dim_feedforward=512):
        super(TAggregate, self).__init__()
        self.num_tokens = 1
        drop_rate = 0.1
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, activation="gelu", dim_feedforward=dim_feedforward, dropout=drop_rate)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + self.num_tokens, embed_dim))
        self.fc = nn.Linear(embed_dim, n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(m.weight, 1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m,  nn.Parameter):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.orthogonal_(m.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)
        pred = self.fc(o[0])
        return pred


# 创建一个随机输入示例
batch_size = 2
clip_length = 10
embed_dim = 64
n_layers = 6
nhead = 8
n_classes = 5

# 创建输入张量
input_tensor = torch.randn(batch_size, clip_length, embed_dim)

# 创建模型实例
model = TAggregate(clip_length=clip_length, embed_dim=embed_dim, n_layers=n_layers, nhead=nhead, n_classes=n_classes)

# 打印输入张量
print("Input tensor Shape:")
print(input_tensor.shape)

# 进行前向传播
output = model(input_tensor)

# 打印输出张量
print("Output tensor Shape:")
print(output.shape)