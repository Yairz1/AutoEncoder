import os

from Utils.data_utils import DataUtils
from Utils.visualization_utils import VisualizationUtils
from architectures.lstm_autoencoder import AutoEncoder

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch
from torch.utils.tensorboard import SummaryWriter

import argparse
from typing import Any

writer = SummaryWriter()
parser = argparse.ArgumentParser(description='lstm_ae_toy')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--clip', type=float, default=1, metavar='N',
                    help='Value to clip the gradient, default 1')
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
    synthetic_data = DataUtils.create_synthetic_data(size=10000, sample_size=50, device_type="cuda:0").cpu()
    VisualizationUtils.visualize_data_examples(synthetic_data,
                                               n=3,
                                               title='Synthetic samples',
                                               xlabel='Time',
                                               ylabel='Value',
                                               path="./plots/synthetic_samples")


# plot_synthetic_samples()

def synthetic_init(path: str):
    data_size = 10000
    series_size = 50
    dataset = DataUtils.create_synthetic_data(size=data_size, sample_size=series_size, device_type=device, path=path)
    dataset = dataset.unsqueeze(2)
    train, val, test = DataUtils.train_val_test_split(dataset, 0.6, 0.2, 0.2)
    train_loader = DataLoader(train, batch_size=args.batch_size)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=len(test))
    auto_encoder = AutoEncoder(input_size=1, hidden_size=args.context_size, num_layers=args.lstm_layers_size)
    auto_encoder = auto_encoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(auto_encoder.parameters(), lr=0.001, momentum=0.9)
    return auto_encoder, train_loader, val_loader, test_loader, criterion, optimizer


def synthetic_train(net: nn.Module,
                    epochs: int,
                    train: Any,
                    val: Any,
                    test: Any,
                    criterion: nn.Module,
                    optimizer: Optimizer):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, inputs in enumerate(train, 0):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(inputs, outputs)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    test_input = next(iter(test))
    test_output = net(test_input)
    loss = criterion(test_input, test_output)
    writer.add_scalar(f"Loss/test ", loss)
    print('Finished Training')
    writer.close()


#
# auto_encoder, train_loader, val_loader, test_loader, criterion, optimizer = synthetic_init(path="./data/synthetic_data")
# synthetic_train(auto_encoder, args.epochs, train_loader, val_loader, test_loader, criterion, optimizer)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
