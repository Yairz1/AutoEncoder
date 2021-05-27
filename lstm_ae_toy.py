import os
from functools import partial

from Utils.data_utils import DataUtils
from Utils.training_utils import TrainingUtils
from Utils.visualization_utils import VisualizationUtils
from Architectures.lstm_autoencoder import AutoEncoder

import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

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

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((auto_encoder.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
    print("Finished Training")


# plot_synthetic_samples()
#
# auto_encoder, train_loader, val_loader, test_loader, criterion, optimizer = init(path="./data/synthetic_data")
# synthetic_train(auto_encoder, args.epochs, train_loader, val_loader, test_loader, criterion, optimizer)

def main(num_samples=1, max_num_epochs=1):
    data_dir = os.path.join("data", "synthetic_data")
    # checkpoint_dir = None
    # config = {"hidden_size": tune.grid_search([10, 20, 30]),
    #           "lr": tune.grid_search([0.001, 0.01, 0.1]),
    #           "grad_clip": tune.grid_search([1, 1.5, 2])}
    config = {"hidden_size": tune.grid_search([10]),
              "lr": tune.grid_search([0.001]),
              "grad_clip": tune.grid_search([1, 1.5])}
    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=max_num_epochs, grace_period=1, reduction_factor=2)
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    result = tune.run(partial(train_synthetic, device=device, data_dir=data_dir),
                      resources_per_trial={"cpu": 1, "gpu": 1},
                      config=config,
                      num_samples=num_samples,
                      scheduler=scheduler,
                      progress_reporter=reporter,
                      name="AE grid search")

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

    best_trained_model = AutoEncoder(input_size=1,
                                     hidden_size=best_trial.config["hidden_size"],
                                     num_layers=args.lstm_layers_size,
                                     device=device)
    best_trained_model.to(device)
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    test_loader, _, _ = DataUtils.load_synthetic_data(data_dir, args.batch_size)
    test_acc = TrainingUtils.test_accuracy(net=best_trained_model,
                                           criterion=None,
                                           test_loader=test_loader,
                                           device=device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10)
