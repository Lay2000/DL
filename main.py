# import all libraries
import string
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from resnet import ResNet18



torch.manual_seed(0)
random.seed(0)

def random_split(dataset, split_size1, split_size2, transform_1=None, transform_2=None):
    indexs = list(range(split_size1 + split_size2))
    indexs_1 = random.sample(indexs, split_size1)
    indexs_1.sort()
    indexs_2 = list(filter(lambda x: x not in indexs_1, indexs))

    dataset_1 = copy.copy(dataset)
    dataset_2 = copy.copy(dataset)

    dataset_1.data = np.array([dataset.data[i] for i in indexs_1])
    dataset_1.targets = [dataset.targets[i] for i in indexs_1]
    dataset_1.transform = transform_1

    dataset_2.data = np.array([dataset.data[i] for i in indexs_2])
    dataset_2.targets = [dataset.targets[i] for i in indexs_2]
    dataset_2.transform = transform_2

    return (dataset_1, dataset_2)

# Training
def train(epoch, net, criterion, trainloader, optimizer, scheduler=None):
    device = 'cuda'
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx+1) % 50 == 0:
          print("iteration : %3d, loss : %0.4f, accuracy : %2.2f" % (batch_idx+1, train_loss/(batch_idx+1), 100.*correct/total))

    if scheduler:
        scheduler.step()
    return train_loss/(batch_idx+1), 100.*correct/total

def test(epoch, net, criterion, testloader):
    device = 'cuda'
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss/(batch_idx+1), 100.*correct/total

def save_checkpoint(net, acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')


def main():
    # Set Config
    parser = argparse.ArgumentParser(description='PyTorch ResNet18 Cifar')
    parser.add_argument('--id', type=str, default='0',
                        help='Experiment ID')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-scheduler', type=str, default=None,
                        help='learning rate sheduler (default: None)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='epochs (default: 15)')
    args = parser.parse_args()

    config = {
    'id': args.id,
    'lr': args.lr,
    'lr_scheduler': args.lr_scheduler,
    'weight_decay': args.weight_decay,
    'epochs': args.epochs
    }
    print(config)

    # Load Data
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    transform_val = transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_size, val_size = 40000, 10000

    wholeset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

    trainset, valset = random_split(wholeset, train_size, val_size, transform_train, transform_val)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=256, shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2)

    for i in range(10):
        print('Label: %5s,  Trainset Samples: %4d,  Valset Samples: %4d,  Testset Samples: %4d' %\
            (classes[i], trainset.targets.count(i), valset.targets.count(i), testset.targets.count(i)))

    # Training Model
    net = ResNet18().to('cuda')
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(net.parameters(), lr=config['lr'],
                        momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) if config['lr_scheduler'] else None

    train_loss_list = []
    train_acc_list = []

    val_loss_list = []
    val_acc_list = []

    for epoch in range(config['epochs']):
        train_loss, train_acc = train(epoch, net, criterion, trainloader, optimizer, scheduler)
        val_loss, val_acc = test(epoch, net, criterion, valloader)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        
        print(("Epoch : %3d, training loss : %0.4f, training accuracy : %2.2f, val loss " + \
        ": %0.4f, val accuracy : %2.2f") % (epoch, train_loss, train_acc, val_loss, val_acc))

    test_loss, test_acc = test(epoch, net, criterion, testloader)
    print("test loss : %0.4f, test accuracy : %2.2f" % (test_loss, test_acc))

    # Visualization & Save Logs

    output_path = './output/'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    dir_path = output_path + config['id'] + '/'

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    plt.figure()
    plt.plot(range(len(train_loss_list)), train_loss_list, 'b')
    plt.plot(range(len(val_loss_list)), val_loss_list, 'r')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("ResNet18 on Cifar-10: Loss vs Number of epochs")
    plt.legend(['train', 'val'])
    plt.savefig(dir_path + 'loss.png')
    plt.savefig(dir_path + 'loss.jpg')

    plt.figure()
    plt.plot(range(len(train_acc_list)), train_acc_list, 'b')
    plt.plot(range(len(val_acc_list)), val_acc_list, 'r')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.title("ResNet18 on Cifar-10: Accuracy vs Number of epochs")
    plt.legend(['train', 'val'])
    plt.savefig(dir_path + 'accuracy.png')
    plt.savefig(dir_path + 'accuracy.jpg')

    logs = [{
    'epoch': i,
    'train_loss': train_loss_list[i],\
    'train_acc': train_acc_list[i],
    'val_loss': val_loss_list[i], 
    'val_acc': val_acc_list[i]
    } for i in range(config['epochs'])
    ]

    df = pd.DataFrame(logs,\
     columns = ['epoch','train_loss','train_acc', 'val_loss', 'val_acc'])
    df.to_csv(dir_path + 'logs.csv')


if __name__ == '__main__':
    main()