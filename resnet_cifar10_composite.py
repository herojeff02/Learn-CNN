import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
import torchvision.transforms as transforms


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @staticmethod
    def test():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    @staticmethod
    def manual():
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    @staticmethod
    def auto():
        return transforms.Compose([
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    @staticmethod
    def none():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


def looper(num_epochs, learning_rate, transform_train, array, block):
    batch_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = './data'

    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=ResNet.test())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ResNet(block, array)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def train():
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print("average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss/len(test_loader.dataset), correct, len(test_loader.dataset), 100. * correct / total, correct, total))
            print('=' * 50)

    for epoch in range(1, num_epochs+1):
        print("epoch : " + str(epoch))
        train()
        if epoch in [30, 50, 100]:
            test()
        scheduler.step()

    model.cpu()
    del model
    del optimizer
    del train_set
    del train_loader
    del criterion
    del scheduler
    del device
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\n\n\n18 auto 0.01")
    looper(2, 0.01, ResNet.auto(), [2, 2, 2, 2], BasicBlock)
    print("\n\n\n18 manual 0.01")
    looper(2, 0.01, ResNet.manual(), [2, 2, 2, 2], BasicBlock)
    print("\n\n\n18 none 0.01")
    looper(2, 0.01, ResNet.none(), [2, 2, 2, 2], BasicBlock)

    print("\n\n\n34 auto 0.01")
    looper(2, 0.01, ResNet.auto(), [3, 4, 6, 3], BasicBlock)
    print("\n\n\n34 manual 0.01")
    looper(2, 0.01, ResNet.manual(), [3, 4, 6, 3], BasicBlock)
    print("\n\n\n34 none 0.01")
    looper(2, 0.01, ResNet.none(), [3, 4, 6, 3], BasicBlock)

    print("\n\n\n50 auto 0.01")
    looper(2, 0.01, ResNet.auto(), [3, 4, 6, 3], Bottleneck)
    print("\n\n\n50 manual 0.01")
    looper(2, 0.01, ResNet.manual(), [3, 4, 6, 3], Bottleneck)
    print("\n\n\n50 none 0.01")
    looper(2, 0.01, ResNet.none(), [3, 4, 6, 3], Bottleneck)

    print("\n\n\n152 auto 0.01")
    looper(2, 0.01, ResNet.auto(), [3, 8, 36, 3], Bottleneck)
    print("\n\n\n152 manual 0.01")
    looper(2, 0.01, ResNet.manual(), [3, 8, 36, 3], Bottleneck)
    print("\n\n\n152 none 0.01")
    looper(2, 0.01, ResNet.none(), [3, 8, 36, 3], Bottleneck)
