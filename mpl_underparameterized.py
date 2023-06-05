from typing import List
import argparse
import pathlib
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from train_utils import (compute_layers, Net)
from utils import load_checkpoint

parser = argparse.ArgumentParser(description='Pytorch Training')
parser.add_argument('--layer_init', default='kaiming', type=str, help='pytorch layer initialization')
parser.add_argument('--varying_layer', default=1, type=int, help='bottleneck layer position')
parser.add_argument('--fixed_layer_values', default=[1, 10, 100, 500, 1000], type=List[int], help='list with number of weights of fixed layer')
parser.add_argument('--params_number', default= 4e+4, type=int, help='total number of model weights')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epcs', default=200, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--output_path', default=pathlib.Path().resolve()+'/results', type=pathlib.Path)
parser.add_argument('--reload', default=False, type=bool, help='warm start model from last epoch')
parser.add_argument('--model_path', default='', type=pathlib.Path)

args = parser.parse_args()

if not args.output_path.exists():
    print('Creating directory:', args.output_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)


#import data set with normalization and data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = args.batch_size

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train, )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# data iterator
dataiter = iter(trainloader)
images, labels = dataiter.next()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#import model
input_size = 3 * 32 * 32


def train(epoch, net, optimizer, criterion):
    net.train()
    loss, total, correct = 0.0 ,0.0 ,0.0

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels) #loss value on batch
        loss.backward()
        optimizer.step()

        loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()   #predicted.eq(targets)

    return loss/total, correct/total


def test(epoch, net, criterion):
    net.eval()
    val_loss, val_total, val_correct = 0.0 ,0.0 ,0.0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    return val_loss/val_total, val_correct/val_total


def train_models(layers, lr, epcs=args.epcs, print_freq=args.epcs//10, model_path=args.warm_start_model_path, reload=args.reload):

    net = Net(layers=layers, initialization=args.layer_init)
    criterion = nn.CrossEntropyLoss()
    t_max = epcs
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=0, last_epoch=- 1, verbose=False)

    if reload:
        net, optimizer, scheduler, start_epc, losses, val_losses, accs, val_accs = load_checkpoint(net, optimizer,
                                                                                                   scheduler,
                                                                                                   filename=model_path)
        net.load_state_dict(torch.load(model_path), strict=False)
        start_epc = int(model_path.split('epcs')[0].split('_')[-1])
        print('** start_epc', start_epc)
    else:
        start_epc = 0
        print('** start_epc', start_epc)
        losses, val_losses = [], []
        accs, val_accs = [], []

    for epoch in range(start_epc,
                       start_epc + epcs):  # loop over the dataset multiple times, number of iterations depends on bs

        loss, acc = train(epoch, net, optimizer, criterion)
        val_loss, val_acc = test(epoch, net, criterion)
        scheduler.step()

        losses.append(loss)
        accs.append(acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if (epoch + 1) % print_freq == 0:
            print('\n epoch', epoch + 1)
            print('accuracy', accs[-1])
            print('val_accuracy', val_accs[-1])

        if (epoch + 1) % print_freq * 2 == 0 or (epoch + 1) % 300 == 0:
            state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'loss': losses,
                     'val_loss': val_losses,
                     'val_acc': val_accs, 'acc': accs}
            torch.save(state,
                       args.output_path + 'cifar10_{}p_{}_{}_{}lr_{}Tmax_{}bs_{}epcs'.format(
                           args.params_number, str(layers), args.layer_init, lr, t_max, batch_size, start_epc + epcs))

    state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'loss': losses,
             'val_loss': val_losses,
             'val_acc': val_accs, 'acc': accs}
    torch.save(state,
               args.output_path + 'cifar10_{}p_{}_{}_{}lr_{}Tmax_{}bs_{}epcs'.format(
                           args.params_number, str(layers), args.layer_init, lr, t_max, batch_size, start_epc + epcs))

    return

for fixed_layer in args.fixed_layer_values:
    layers = compute_layers(params_number=args.params_number, varying_layer=1,  input_size=input_size, fixed_layer_size=fixed_layer, output_size=output_size)
    train_models(layers=layers, lr=args.lr, epcs=args.epcs, print_freq=args.epcs//10, model_path=args.warm_start_model_path, reload=args.reload)



