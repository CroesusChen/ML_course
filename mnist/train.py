import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# # 检查是否可以利用GPU
# train_on_gpu = torch.cuda.is_available()
#
# if not train_on_gpu:
#     print('CUDA is not available.')
# else:
#     print('CUDA is available!')

# number of subprocesses to use for data loading
num_workers = 0
# 每批加载16张图片
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2
epochs=30
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
class_p = list(0. for i in range(10))

# 将数据转换为torch.FloatTensor，并标准化。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# 选择训练集与测试集的数据
train_data = datasets.MNIST('data', train=True,
                              download=True, transform=transform)
test_data = datasets.MNIST('data', train=False,
                             download=True, transform=transform)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    shuffle = True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model,  test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction = 'sum').item()  # sum up batch loss
            # pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
            _, pred = torch.max(output, 1)
            preds = np.squeeze(pred.cpu().numpy())
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            for i in range(batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
                pre_label = preds[i]
                class_p[pre_label] += 1
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)\t precision : %2d%% (%2d/%2d)' % (
                i, 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i]), 100 * class_correct[i] / class_p[i],
                np.sum(class_correct[i]), np.sum(class_p[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (i))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)\t precision : %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total), 100. * np.sum(class_correct) / np.sum(class_p),
        np.sum(class_correct), np.sum(class_p)))


use_cuda = True
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(1, epochs + 1):
    train( model, train_loader, optimizer, epoch)
# model.load_state_dict(torch.load("mnist_cnn.pt"))
test( model, test_loader)

torch.save(model.state_dict(), "mnist_cnn.pt")







