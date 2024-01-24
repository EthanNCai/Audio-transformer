import torch
import torch.nn as nn

x = torch.randn((32, 1, 88200))
print(x.shape)

patch_size = 180
x = x.view(x.shape[0], patch_size, -1)
print(x.shape)


ll = nn.LazyLinear(512)

x = ll(x)
print(x.shape)
