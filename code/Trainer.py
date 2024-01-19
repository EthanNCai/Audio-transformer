from Model import FCNet, DownSampling
from Dataset import UrbanSound8K
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


unified_sample_rate = 22050
unified_sec = 4
classes = 10
learning_rate = 0.001
epochs = 10

dataset_train = UrbanSound8K(dataset_root='../data/UrbanSound8k', unified_sample_rate=unified_sample_rate,
                             unified_sec=unified_sec,
                             mode='train',
                             train_ratio=0.8)

dataset_test = UrbanSound8K(dataset_root='../data/UrbanSound8k', unified_sample_rate=unified_sample_rate,
                            unified_sec=unified_sec,
                            mode='test',
                            train_ratio=0.8)

train_loader = DataLoader(dataset=dataset_train, batch_size=31, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=31, shuffle=False, drop_last=True)

model = DownSampling(unified_sample_rate * unified_sec)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(epochs):
    for i, (audio_wave, labels) in enumerate(train_loader):
        # forward
        outputs = model(audio_wave)
        loss = criterion(outputs, labels)

        # update the weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, total_step,
                                                                     loss.item()))
        break
    break

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for audio_wave, labels in test_loader:
        outputs = model(audio_wave)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('测试集准确率: {:.2f}%'.format(100 * correct / total))