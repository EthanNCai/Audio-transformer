import torch
import torch.nn as nn
import numpy as np


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
        self.fc1 = LazyLinear(num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class EmbeddingFC(nn.Module):
    def __init__(self, output_size, *args, **kwargs):
        super(EmbeddingFC, self).__init__(*args, **kwargs)

        self.linear1 = LazyLinear(output_size)
        self.bn = nn.BatchNorm1d(output_size)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x_):
        out = x_.view(x_.size(0), -1)
        x_ = self.linear1(x_)
        x_ = self.bn(x_)
        x_ = self.relu(x_)
        return x_


class Transformer(nn.Module):
    def __init__(self, embedded_size):
        super().__init__()
        self.enc_layer = nn.TransformerEncoderLayer(d_model=embedded_size, nhead=8, activation="gelu",
                                                    dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(self.enc_layer, num_layers=6, norm=nn.LayerNorm(512))

    def forward(self, x):
        encoded_src = self.transformer_enc(x)
        return encoded_src


class LazyLinear(nn.Module):
    def __init__(self, output_dim):
        super(LazyLinear, self).__init__()
        self.output_dim = output_dim
        self.input_dim = None
        self.fc = None

    def forward(self, x):
        if self.fc is None:
            print('here')
            self.input_dim = x.size(1)
            self.fc = nn.Linear(self.input_dim, self.output_dim)
        return self.fc(x)


class ChickenNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 64
        self.audio_wave_len = 88200
        self.embedded_feature_size = 512
        self.num_classes = 10
        self.feature_size = None
        self.transformer_module = Transformer(self.embedded_feature_size)
        self.down_module = DownSampling()
        self.fc_module = FC(self.num_classes)
        self.mfc_module = EmbeddingFC(self.embedded_feature_size)
        self.test_linear = nn.Linear(11008, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.down_module(x)
        """
        if self.feature_size is None:
            total_len = x.numel() // self.batch_size
            for i in range(128, 0, -1):
                if total_len % i == 0:
                    self.feature_size = i
                    break
        x = x.view(self.batch_size, -1, self.feature_size)
        batch_size, seq_len, raw_feature_size = x.shape
        x = x.view(-1, raw_feature_size)
        x = self.mfc_module(x)
        x = x.view(batch_size, seq_len, self.embedded_feature_size)
        # x = self.transformer_module(x)
        """
        x = x.view(x.shape[0], -1)
        x = self.test_linear(x)
        return x


def test():
    # sup-parameters
    batch_size = 64
    audio_wave_len = 88200
    embedded_feature_size = 512
    num_classes = 10
    feature_size = None

    # modules
    down_module = DownSampling()
    transformer_module = Transformer(embedded_feature_size)
    fc_module = FC(num_classes)
    mfc_module = EmbeddingFC(embedded_feature_size)

    # forward
    x = torch.rand((batch_size, audio_wave_len))
    x = x.unsqueeze(1)
    x = down_module(x)
    if feature_size is None:
        total_len = x.numel() // batch_size
        for i in range(128, 0, -1):
            if total_len % i == 0:
                feature_size = i
                break
    x = x.view(batch_size, -1, feature_size)
    batch_size, seq_len, raw_feature_size = x.shape
    x = mfc_module(x)
    x = x.view(batch_size, seq_len, embedded_feature_size)
    x = transformer_module(x)
    x = fc_module(x)

    print(x.shape)


def test2():
    x = torch.rand((32, 88200))
    print(x.shape)
    chicken_net = ChickenNet()
    x = chicken_net(x)
    print(x.shape)
