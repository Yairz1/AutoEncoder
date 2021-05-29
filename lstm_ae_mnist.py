import os
from functools import partial

from matplotlib import pyplot as plt
from torch import nn

from Utils.data_utils import DataUtils
from Utils.parameters_tune import ParameterTuning
from Utils.training_utils import TrainingUtils

import torch

import argparse

parser = argparse.ArgumentParser(description='lstm_ae_toy')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lstm-layers-size', type=int, default=3, metavar='N',
                    help='lstm layers number, default 3')
parser.add_argument('--lstm-dropout', type=int, default=0.2, metavar='N',
                    help='lstm layers number, default 0')
parser.add_argument('--optimizer', type=str, default="adam", metavar='N',
                    help='optimizer, default adam')
parser.add_argument('--load', type=bool, default=True, metavar='N',
                    help='To load or create new data, default True')
parser.add_argument('--input-size', type=int, default=1, metavar='N',
                    help='LSTM feature input size, default 1')
parser.add_argument('--seq-len', type=int, default=784, metavar='N',
                    help='LSTM sequence series length, default 784')
args = parser.parse_args()
print(torch.cuda.get_device_name(0))


def plot_mnist(path, n, loader):
    images, labels = next(iter(loader))
    fig, axs = plt.subplots(n)
    for i, ax in enumerate(axs):
        dim = int(images.shape[1] ** 0.5)
        ax.imshow(images[i].reshape(dim, dim), cmap="gray")
        ax.set_title(f"Digit {labels[i]}")
    fig.tight_layout(pad=0.5)
    fig.show()
    if path:
        fig.savefig(path)


def compare_mnist_reconstruction(device, test_loader, model, path):
    with torch.no_grad():
        pass


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # plots_suffix = os.path.join("plots", "job_plots")
    plots_suffix = os.path.join("plots", "mnist")
    data_dir = os.path.join("data")
    config = {"hidden_size": [256],
              "lr": [0.001],
              "grad_clip": [1, None]}
    test_loader, train_loader, _ = DataUtils.data_loader_factory("mnist", data_dir, args.batch_size, True)
    plot_mnist(path=os.path.join(plots_suffix, "example"), n=3, loader=train_loader)
    criterion = nn.CrossEntropyLoss()
    tune = ParameterTuning(config_options=config)
    tune.run(train_func=partial(TrainingUtils.train,
                                input_size=args.input_size,
                                input_seq_size=args.seq_len,
                                dataset_name="mnist",
                                batch_size=args.batch_size,
                                criterion=criterion,
                                optimizer=args.optimizer,
                                lstm_layers_size=args.lstm_layers_size,
                                epochs=args.epochs,
                                load_data=args.load,
                                device=device,
                                data_dir=data_dir),
             test_func=partial(TrainingUtils.test_accuracy,
                               criterion=criterion,
                               test_loader=test_loader,
                               device=device))

    compare_mnist_reconstruction(device, test_loader, tune.best_model, os.path.join(plots_suffix, "reconstruct"))
    print("Best trial config: {}".format(tune.best_config))
    print("Best trial final validation loss: {}".format(round(tune.get_best_val_loss(), 3)))
    print("Best trial test set accuracy: {}".format(round(tune.best_loss, 3)))
    tune.plot_validation_trails(path=os.path.join(plots_suffix, "all_validation_trails"))
    tune.plot_train_trails(path=os.path.join(plots_suffix, "all_train_trails"))
    tune.plot_best_train(path=os.path.join(plots_suffix, "best train trail"))
    tune.plot_best_val(path=os.path.join(plots_suffix, "best validation trail"))


if __name__ == "__main__":
    main()
