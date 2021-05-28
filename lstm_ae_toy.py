import os
from functools import partial

from Utils.data_utils import DataUtils
from Utils.parameters_tune import ParameterTuning
from Utils.training_utils import TrainingUtils
from Utils.visualization_utils import VisualizationUtils
from Architectures.lstm_autoencoder import AutoEncoder

import torch.optim as optim
import torch.nn as nn
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


def plot_synthetic_samples():
    synthetic_data = DataUtils.create_synthetic_data(size=10000,
                                                     sample_size=50,
                                                     device_type="cpu",
                                                     path="./data/synthetic_data",
                                                     load=False)
    VisualizationUtils.visualize_data_examples(synthetic_data,
                                               n=3,
                                               title='Synthetic samples',
                                               xlabel='Time',
                                               ylabel='Value',
                                               path="./plots/synthetic_samples")


# plot_synthetic_samples()

def init(hidden_size: int, path: str, checkpoint_dir: str, device: Any):
    test_loader, train_loader, val_loader = DataUtils.load_synthetic_data(path, args.batch_size)
    auto_encoder = AutoEncoder(input_size=1, hidden_size=hidden_size, num_layers=args.lstm_layers_size, device=device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(auto_encoder.parameters(), lr=0.001, momentum=0.9)
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        auto_encoder.load_state_dict(model_state)
        auto_encoder = auto_encoder.to(device)
        optimizer.load_state_dict(optimizer_state)
    return auto_encoder, train_loader, val_loader, test_loader, criterion, optimizer


def train_synthetic(config, device, checkpoint_dir=None, data_dir=None):
    auto_encoder, train_loader, val_loader, test_loader, criterion, optimizer = init(config["hidden_size"],
                                                                                     data_dir,
                                                                                     checkpoint_dir,
                                                                                     device)
    auto_encoder.to(device)
    training_info = []
    val_info = []
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            data = data.to(device)
            optimizer.zero_grad()
            outputs = auto_encoder(data)
            loss = criterion(outputs, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(auto_encoder.parameters(), config['grad_clip'])
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
        training_info.append(loss.item())

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                data = data.to(device)
                outputs = auto_encoder(data)
                loss = criterion(outputs, data)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        val_info.append(val_loss / val_steps)
    print("Finished Training")
    return auto_encoder, training_info, val_info


def main():
    # plot_synthetic_samples()

    data_dir = os.path.join("data", "synthetic_data")
    # config = {"hidden_size": [10, 20, 30],
    #           "lr": [0.001, 0.01, 0.1],
    #           "grad_clip": [1, 1.5, 2]}
    config = {"hidden_size": [10],
              "lr": [0.001],
              "grad_clip": [1, 1.5]}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_loader, _, _ = DataUtils.load_synthetic_data(data_dir, args.batch_size)

    tune = ParameterTuning(config_options=config)
    tune.run(train_func=partial(train_synthetic, device=device, data_dir=data_dir),
             test_func=partial(TrainingUtils.test_accuracy, test_loader=test_loader, device=device))

    print("Best trial config: {}".format(tune.best_config))
    print("Best trial final validation loss: {}".format(round(tune.get_best_val_loss(), 3)))
    print("Best trial test set accuracy: {}".format(round(tune.best_loss, 3)))
    tune.plot_validation_trails(path="")


if __name__ == "__main__":
    main()
