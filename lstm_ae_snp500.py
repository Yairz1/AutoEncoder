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


def plot_stock_high_prices(path):
    amazon_daily_max, googl_daily_max = DataUtils.load_snp500_amzn_google_daily_max(path)
    VisualizationUtils.plot_df_columns(amazon_daily_max,
                                       "date",
                                       "high",
                                       "Amazon \nTime vs Daily Maximum",
                                       "Time",
                                       "Daily high")
    VisualizationUtils.plot_df_columns(googl_daily_max,
                                       "date",
                                       "high",
                                       "Google \nTime vs Daily Maximum",
                                       "Time",
                                       "Daily high")


def main():
    path = os.path.join("data", "SP 500 Stock Prices 2014-2017.csv")
    plot_stock_high_prices(path)
    data_dir = os.path.join("data", "SP 500 Stock Prices 2014-2017.csv")
    # plots_suffix = os.path.join("plots", "job_plots")
    plots_suffix = os.path.join("plots")
    config = {"hidden_size": [256],
              "lr": [0.001],
              "grad_clip": [None]}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_loader, _, _ = DataUtils.load_snp500(data_dir, args.batch_size)
    criterion = nn.MSELoss()
    tune = ParameterTuning(config_options=config)
    tune.run(train_func=partial(TrainingUtils.train,
                                auto_encoder_init=ToyAutoEncoder,
                                input_size=1,
                                input_seq_size=1007,
                                dataset_name="sp500",
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

    # compare_reconstruction(device, test_loader, tune.best_model, os.path.join(plots_suffix, "reconstruct"))
    print("Best trial config: {}".format(tune.best_config))
    print("Best trial final validation loss: {}".format(round(tune.get_best_val_loss(), 3)))
    print("Best trial test set accuracy: {}".format(round(tune.best_loss, 3)))
    tune.plot_validation_trails(path=os.path.join(plots_suffix, "all_validation_trails"))
    tune.plot_train_trails(path=os.path.join(plots_suffix, "all_train_trails"))
    tune.plot_best_train(path=os.path.join(plots_suffix, "best train trail"))
    tune.plot_best_val(path=os.path.join(plots_suffix, "best validation trail"))


if __name__ == "__main__":
    main()
