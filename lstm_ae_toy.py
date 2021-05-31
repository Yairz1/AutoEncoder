import os
from functools import partial

from torch import nn

from Architectures.lstm_autoencoder import ToyAutoEncoder
from Utils.data_utils import DataUtils
from Utils.parameters_tune import ParameterTuning
from Utils.training_utils import TrainingUtils
from Utils.visualization_utils import VisualizationUtils

import torch
from torch.utils.tensorboard import SummaryWriter

import argparse

writer = SummaryWriter()
parser = argparse.ArgumentParser(description='lstm_ae_toy')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lstm-layers-size', type=int, default=3, metavar='N',
                    help='lstm layers number, default 3')
parser.add_argument('--lstm-dropout', type=int, default=0.2, metavar='N',
                    help='lstm layers number, default 0')
parser.add_argument('--optimizer', type=str, default="adam", metavar='N',
                    help='optimizer, default adam')
parser.add_argument('--load', type=bool, default=True, metavar='N',
                    help='To load or create new data, default True')
parser.add_argument('--decoder-output-size', type=int, default=256, metavar='N',
                    help='LSTM size at the end, default 256')
args = parser.parse_args()
print(torch.cuda.get_device_name(0))


def plot_synthetic_samples(path, data_dir):
    synthetic_data = DataUtils.create_synthetic_data(size=10000,
                                                     sample_size=50,
                                                     device_type="cpu",
                                                     path=data_dir,
                                                     load=False)
    VisualizationUtils.visualize_data_examples(synthetic_data,
                                               n=3,
                                               title='Synthetic samples',
                                               xlabel='Time',
                                               ylabel='Value',
                                               path=path)


def main():
    data_dir = os.path.join("data", "synthetic_data")
    # plots_suffix = os.path.join("plots", "job_plots")
    plots_suffix = os.path.join("plots")
    plot_synthetic_samples(os.path.join(plots_suffix, "synthetic_data_examples"), data_dir)
    config = {"hidden_size": [40, 256],
              "lr": [0.01, 0.001],
              "grad_clip": [1, None]}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_loader, _, _ = DataUtils.load_synthetic_data(data_dir, args.batch_size, args.load)
    criterion = nn.MSELoss()
    tune = ParameterTuning(config_options=config)
    tune.run(train_func=partial(TrainingUtils.train,
                                auto_encoder_init=ToyAutoEncoder,
                                input_size=1,
                                input_seq_size=50,
                                dataset_name="synthetic_data",
                                batch_size=args.batch_size,
                                criterion=criterion,
                                optimizer=args.optimizer,
                                lstm_layers_size=args.lstm_layers_size,
                                decoder_output_size=args.decoder_output_size,
                                epochs=args.epochs,
                                load_data=args.load,
                                training_iteration=TrainingUtils.training_iteration,
                                validation=TrainingUtils.validation,
                                device=device,
                                data_dir=data_dir),
             test_func=partial(TrainingUtils.test_accuracy,
                               criterion=criterion,
                               test_loader=test_loader,
                               device=device))

    VisualizationUtils.compare_reconstruction(device,
                                              test_loader,
                                              tune.best_model,
                                              os.path.join(plots_suffix, "reconstruct"))
    print("Best trial config: {}".format(tune.best_config))
    print("Best trial final validation loss: {}".format(round(tune.get_best_val_loss(), 3)))
    print("Best trial test set accuracy: {}".format(round(tune.best_loss, 3)))
    tune.plot_validation_trails(path=os.path.join(plots_suffix, "all_validation_trails"))
    tune.plot_train_trails(path=os.path.join(plots_suffix, "all_train_trails"))
    tune.plot_best_train(path=os.path.join(plots_suffix, "best train trail"))
    tune.plot_best_val(path=os.path.join(plots_suffix, "best validation trail"))


if __name__ == "__main__":
    main()
