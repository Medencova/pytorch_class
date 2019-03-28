'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import random

import os
import argparse
import re
import cv2
import operator
import numpy
from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(150),
    # transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda img: transforms.functional.adjust_hue(img, random.uniform(-0.5, 0.5))),
    # transforms.Lambda(lambda img: transforms.functional.adjust_contrast(img, 2)),
    # transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# transform_train.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(150),
    # transforms.Lambda(lambda img: transforms.functional.adjust_hue(img, random.uniform(-0.5, 0.5))),
    # transforms.Lambda(lambda img: transforms.functional.adjust_contrast(img, 2)),
    # transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transform_predict = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(150),
    # transforms.Lambda(lambda img: transforms.functional.adjust_contrast(img, 2)),
    # transforms.Lambda(lambda img: transforms.functional.adjust_hue(img, random.uniform(-0.5, 0.5))),
    # transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class CellsDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        entry = self.data[item]
        image = cv2.imread(entry['path'])
        label = entry['label']
        return dict(
            label=label,
            image=self.transform(image),
        )


def getdata(txt):
    data = []
    f = open(txt, 'r')
    for line in f:
        lab = ('py', 'ar', 'ec', 'pn').index(re.search(r'/.._', line).group(0)[1:3])
        print(line)
        data.append({'path': line[:-1], 'label': lab})
    return data


trainset = CellsDataset(data=getdata('./train2.txt'), transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, pin_memory=True, num_workers=2)

testset = CellsDataset(data=getdata('./test2.txt'), transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, pin_memory=True, num_workers=2)

predictset = CellsDataset(data=getdata('./predict.txt'), transform=transform_predict)
predictloader = torch.utils.data.DataLoader(predictset, batch_size=8, shuffle=False, pin_memory=True, num_workers=1)

classes = ('py', 'ar', 'ec', 'pn')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet101()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = ShuffleNetV2(1)
net = net.to(device)

writer = SummaryWriter()


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7.crop')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, entry in enumerate(trainloader):
        inputs = entry['image']
        targets = entry['label']
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # print (inputs)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx % 10 == 0:
            writer.add_scalar('train/Loss-batch', train_loss / (batch_idx + 1), batch_idx)
        last = batch_idx
    writer.add_scalar('train/Loss-epoch', train_loss/(last+1), epoch)
    writer.add_scalar('train/Acc-epoch-%', 100.*correct/total, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, entry in enumerate(testloader):
            inputs = entry['image']
            targets = entry['label']

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            last = batch_idx
    writer.add_scalar('valid/Loss-epoch', test_loss / (last + 1), epoch)
    writer.add_scalar('valid/Acc-epoch-%', 100. * correct / total, epoch)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7.crop')
        best_acc = acc


def predict():
    # with torch.no_grad():
    big_predict = []
    target_arr = numpy.array([])
    count = 5
    length = 0
    flag = 0
    for i in range(0, count):
        pr = numpy.array([])

        for batch_idx, entry in enumerate(predictloader):
            inputs = entry['image']
            targets = entry['label']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            predicted = outputs.max(1)
            pr = numpy.concatenate([pr, predicted[1].cpu().numpy()])
            if flag == 0:
                target_arr = numpy.concatenate((target_arr, targets.cpu().numpy()))
        flag = 1
        big_predict.append(pr)
        length = int(len(big_predict[0]))


    result = []
    for image in range(0, length):
        dict = {}
        for batch in range(0, count):
            if not dict:
                dict[big_predict[batch][image]] = 1
                continue
            if str(big_predict[batch][image]) in dict:
                dict[big_predict[batch][image]] += 1
            else:
                dict[big_predict[batch][image]] = 1
        result.append(int(max(dict, key=dict.get)))

    print(length)
    print(target_arr)
    print(result)




for epoch in range(start_epoch, start_epoch+750):
    train(epoch)
    test(epoch)

# predict()
writer.close()
