import torch

print("cat1")
x = torch.randn(2, 3)
print(x.shape)
y = torch.cat((x, x, x), dim=0)
print(y.shape)


print("cat2")
x = torch.randn(2, 3)
print(x.shape)
y = torch.cat((x, x, x), dim=1)
print(y.shape)


print("expand")
x = torch.randn((1, 1, 31))
print(x.shape)
y = x.expand(33, -1, -   1)
print(y.shape)

