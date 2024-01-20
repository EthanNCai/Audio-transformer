import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTTransformer(nn.Module):
    def __init__(self, d_model, num_classes, num_layers, nhead, dim_feedforward, dropout):
        super(MNISTTransformer, self).__init__()
        self.embedding = nn.Linear(2, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        x = x.permute(0, 2, 1)  # (batch_size, d_model, seq_length)
        output = self.transformer_encoder(x)  # (batch_size, d_model, seq_length)
        output = output.mean(dim=2)  # (batch_size, d_model)
        output = self.fc(output)  # (batch_size, num_classes)
        return output

# 构建模型
d_model = 128
num_classes = 10
num_layers = 2
nhead = 4
dim_feedforward = 256
dropout = 0.2

model = MNISTTransformer(d_model, num_classes, num_layers, nhead, dim_feedforward, dropout)

# 构建数据准备
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载和加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# 数据加载器
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义优化器和损失函数
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 训练模型
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.view(-1, 2)  # 将每两个像素作为一个令牌
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播和计算损失
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 2)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')