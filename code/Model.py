import torch
import torch.nn as nn
import math


class DownSampling(nn.Module):
    def __init__(self):
        super(DownSampling, self).__init__()
        self.dilation = 2
        self.kernel_size = 3
        self.stride = 2

        layers = [
            nn.Conv1d(1, 2, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(2, 4, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(4, 8, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, True),

        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class FC(nn.Module):
    def __init__(self, num_classes):
        super(FC, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class EmbeddingFC(nn.Module):
    def __init__(self, output_size, *args, **kwargs):
        super(EmbeddingFC, self).__init__(*args, **kwargs)

        self.linear1 = nn.LazyLinear(output_size)
        self.bn = nn.BatchNorm1d(output_size)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x_):
        out = x_.view(x_.size(0), -1)
        x_ = self.linear1(x_)
        x_ = self.bn(x_)
        x_ = self.relu(x_)
        return x_


class UnifiedFC(nn.Module):
    def __init__(self, out_shape):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(out_shape)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Transformer(nn.Module):
    def __init__(self, embedded_size, num_heads):
        super().__init__()
        self.enc_layer = nn.TransformerEncoderLayer(d_model=embedded_size, nhead=num_heads, activation="gelu",
                                                    dim_feedforward=4096, dropout=0.1, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(self.enc_layer, num_layers=6, norm=nn.LayerNorm(embedded_size))

    def forward(self, x):
        x = self.transformer_enc(x)
        return x


class ChickenNet(nn.Module):

    def __init__(self):
        super().__init__()
        # train params
        self.batch_size = 32
        self.wave_size = 88200
        self.num_classes = 10
        self.patch_size = 100

        # transformer params
        self.num_heads = 8
        self.embedded_size = 512
        self.num_tokens = self.wave_size // self.patch_size
        assert self.wave_size % self.patch_size == 0, f'num_heads {self.patch_size} not divisible'
        assert self.embedded_size % self.num_heads == 0, f'patch size {self.patch_size} not divisible'
        # assert (self.wave_size // self.patch_size) < 512, f'seq len {self.wave_size // self.patch_size} too large'

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedded_size))
        self.pos_emb1D = nn.Parameter(torch.randn(self.num_tokens + 1, self.embedded_size))
        self.clf_mlp = nn.Linear(self.embedded_size, self.num_classes)
        self.emb_mlp = nn.LazyLinear(self.embedded_size)
        self.transformer_encoder = Transformer(self.embedded_size, self.num_heads)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.down_module = DownSampling()

    def get_batch_expanded_cls_token(self):
        return self.cls_token.expand([self.batch_size, -1, -1])

    def forward(self, x_in):
        patched_x = x_in.view(x_in.shape[0], -1, self.patch_size)
        # print(patched_x.shape) -> torch.Size([64, 490, 180])

        embedded_x = self.emb_mlp(patched_x)
        embedded_x = torch.cat(
            (self.get_batch_expanded_cls_token(), embedded_x), dim=1)
        # print(embedded_x.shape) -> torch.Size([64, 491, 128])
        # print(self.pos_emb1D.shape) -> torch.Size([491, 128])

        pos_encoded_x = self.dropout(embedded_x + self.pos_emb1D)
        # print(pos_encoded_x.shape) -> torch.Size([64, 491, 128])

        encoded_x = self.transformer_encoder(pos_encoded_x)
        result = self.clf_mlp(encoded_x[:, 0, :])

        return result


def test():
    model = ChickenNet()
    x = torch.randn(64, 1, 88200)
    x = model(x)


