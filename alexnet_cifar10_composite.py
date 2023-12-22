import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=0.5)
        out = F.log_softmax(self.fc3(out), dim=1)
        return out

    @staticmethod
    def test():
        return transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    @staticmethod
    def manual():
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    @staticmethod
    def auto():
        return transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    @staticmethod
    def none():
        return transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


def looper(num_epochs, learning_rate, transform_train):
    batch_size = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = './data'

    train_set = datasets.CIFAR10(root=root, train=True, transform=transform_train, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.CIFAR10(root=root, train=False, transform=AlexNet.test(), download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = AlexNet().to(device)
    criterion = F.nll_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train():
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print("average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
            print('=' * 50)

    for epoch in range(1, num_epochs + 1):
        print("epoch : " + str(epoch))
        train()
        if epoch in [30, 50, 100]:
            test()

    model.cpu()
    del model
    del optimizer
    del train_set
    del train_loader
    del criterion
    del device
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\n\n\nauto 0.01")
    looper(100, 0.01, AlexNet.auto())
    print("\n\n\nmanual 0.01")
    looper(100, 0.01, AlexNet.manual())
    print("\n\n\nnone 0.01")
    looper(100, 0.01, AlexNet.none())

    print("\n\n\nauto 0.0001")
    looper(100, 0.0001, AlexNet.auto())
    print("\n\n\nmanual 0.0001")
    looper(100, 0.0001, AlexNet.manual())
    print("\n\n\nnone 0.0001")
    looper(100, 0.0001, AlexNet.none())
