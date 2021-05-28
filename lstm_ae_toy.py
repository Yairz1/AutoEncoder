import os
from functools import partial

from Utils.data_utils import DataUtils
from Utils.parameters_tune import ParameterTuning
from Utils.training_utils import TrainingUtils
from Utils.visualization_utils import VisualizationUtils

import torch
from torch.utils.tensorboard import SummaryWriter

import argparse

writer = SummaryWriter()
parser = argparse.ArgumentParser(description='lstm_ae_toy')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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
    tune.run(train_func=partial(TrainingUtils.train_synthetic,
                                batch_size=args.batch_size,
                                lstm_layers_size=args.lstm_layers_size,
                                epochs=args.epochs,
                                device=device,
                                data_dir=data_dir),
             test_func=partial(TrainingUtils.test_accuracy, test_loader=test_loader, device=device))

    print("Best trial config: {}".format(tune.best_config))
    print("Best trial final validation loss: {}".format(round(tune.get_best_val_loss(), 3)))
    print("Best trial test set accuracy: {}".format(round(tune.best_loss, 3)))
    tune.plot_validation_trails(path="")
    tune.plot_train_trails(path="")
    tune.plot_best_train(path="")
    tune.plot_best_val(path="")


if __name__ == "__main__":
    main()
