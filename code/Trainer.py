import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Model import FC, DownSampling, ChickenNet
from Dataset import UrbanSound8K

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
unified_sample_rate = 22050
unified_sec = 4
classes = 10
learning_rate = 0.0002
epochs = 30

dataset_train = UrbanSound8K(dataset_root='../data/UrbanSound8k', unified_sample_rate=unified_sample_rate,
                             unified_sec=unified_sec,
                             mode='train',
                             train_ratio=0.9)

dataset_test = UrbanSound8K(dataset_root='../data/UrbanSound8k', unified_sample_rate=unified_sample_rate,
                            unified_sec=unified_sec,
                            mode='test',
                            train_ratio=0.9)

train_loader = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False, drop_last=True)

model = ChickenNet()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam((model.parameters()), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(epochs):
    for i, (input_, labels) in enumerate(train_loader):

        input_ = input_.unsqueeze(1).to(device)
        input_ = input_.to(device)
        labels = labels.to(device)

        outputs = model(input_)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\rEpoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, total_step,
                                                                    loss.item()), end='')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for input_, labels in test_loader:

        input_ = input_.unsqueeze(1).to(device)
        input_ = input_.to(device)
        labels = labels.to(device)

        outputs = model(input_)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy: {:.2f}%'.format(100 * correct / total))
