from Utils.data_utils import DataUtils
from Utils.visualization_utils import VisualizationUtils
from architectures.lstm_autoencoder import AutoEncoder

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

import argparse
from typing import Any

parser = argparse.ArgumentParser(description='lstm_ae_toy')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--context-size', type=int, default=25, metavar='N',
                    help='Context vector size, default 25')
parser.add_argument('--lstm-layers-size', type=int, default=3, metavar='N',
                    help='lstm layers number, default 3')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


def plot_synthetic_samples():
    synthetic_data = DataUtils.create_synthetic_data(size=100000,
                                                     sample_size=50,
                                                     device_type="cuda:0",
                                                     path="./data/synthetic_data").cpu()
    VisualizationUtils.visualize_data_examples(synthetic_data,
                                               n=3,
                                               title='Synthetic samples',
                                               xlabel='Time',
                                               ylabel='Value',
                                               path="./plots/synthetic_samples")


# plot_synthetic_samples()

def synthetic_init():
    data_size = 100000
    series_size = 50
    dataset = DataUtils.create_synthetic_data(size=data_size,
                                              sample_size=series_size,
                                              device_type=device,
                                              path="./data/synthetic_data")
    train, val, test = DataUtils.train_val_test_split(dataset, 0.6, 0.2, 0.2)
    train_loader = DataLoader(train, batch_size=args.batch_size)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    auto_encoder = AutoEncoder(input_size=series_size, hidden_size=args.context_size, num_layers=args.lstm_layers_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(auto_encoder.parameters(), lr=0.001, momentum=0.9)
    return auto_encoder, train_loader, val_loader, test_loader, criterion, optimizer


def synthetic_train(net: nn.Module, train: Any, val: Any, test: Any, criterion: nn.Module, optimizer: Optimizer):
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


auto_encoder, train_loader, val_loader, test_loader, criterion, optimizer = synthetic_init()
synthetic_train(auto_encoder, train_loader, val_loader, test_loader, criterion, optimizer)
