import os
from functools import partial

from torch import nn

from Architectures.lstm_autoencoder import ToyAutoEncoder, SP500AutoEncoder, SP500AutoEncoder_prediction
from Utils.data_utils import DataUtils
from Utils.parameters_tune import ParameterTuning
from Utils.training_utils import TrainingUtils
from Utils.visualization_utils import VisualizationUtils

import torch
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np

writer = SummaryWriter()
parser = argparse.ArgumentParser(description='lstm_ae_snp500')
parser.add_argument('--batch-size', type=int, default=40, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',  # 100,150
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
parser.add_argument('--folds', type=int, default=1, metavar='N',
                    help='k for k fold')
args = parser.parse_args()
print(torch.cuda.get_device_name(0))


def plot_stock_high_prices(path, plots_suffix):
    amazon_daily_max, googl_daily_max = DataUtils.load_snp500_amzn_google_daily_max(path)
    VisualizationUtils.plot_df_columns(amazon_daily_max,
                                       "date",
                                       "high",
                                       "Amazon \nTime vs Daily Maximum",
                                       "Time",
                                       "Daily high",
                                       os.path.join(plots_suffix, "amazon"))
    VisualizationUtils.plot_df_columns(googl_daily_max,
                                       "date",
                                       "high",
                                       "Google \nTime vs Daily Maximum",
                                       "Time",
                                       "Daily high",
                                       os.path.join(plots_suffix, "google"))


def snp500_reconstruct(data_tensor, config, device, plots_suffix, n_part):
    train_idxs, test_idxs = DataUtils.create_random_train_test_indices_split(data_tensor.shape[0], 0.85, 0.15)
    data_gen = DataUtils.generate_random_split(data_tensor[train_idxs], args.folds, 0.9, 0.1)
    test_loader = DataUtils.create_data_loader(data_tensor[test_idxs].unsqueeze(2), args.batch_size)
    criterion = nn.MSELoss()
    tune = ParameterTuning()
    tune.kfold_run(train_func=partial(TrainingUtils.kfold_train,
                                      # auto_encoder_init=ToyAutoEncoder,
                                      auto_encoder_init=SP500AutoEncoder,
                                      lr=config["lr"],
                                      hidden_size=config["hidden_size"],
                                      input_size=1,
                                      input_seq_size=data_tensor.shape[1],  # 1007
                                      batch_size=args.batch_size,
                                      criterion=criterion,
                                      optimizer=args.optimizer,
                                      lstm_layers_size=args.lstm_layers_size,
                                      decoder_output_size=args.decoder_output_size,
                                      epochs=args.epochs,
                                      training_iteration=TrainingUtils.training_iteration,
                                      validation=TrainingUtils.validation,
                                      device=device),
                   test_func=partial(TrainingUtils.test_accuracy,
                                     criterion=criterion,
                                     test_loader=test_loader,
                                     device=device),
                   data_tensor=data_tensor[train_idxs].unsqueeze(2),
                   data_generator=data_gen,
                   batch_size=args.batch_size,
                   config=config)

    test_loader = DataUtils.create_data_loader(data_tensor[test_idxs].unsqueeze(2),
                                               min(len(test_idxs), args.batch_size))

    test_input = next(iter(test_loader))
    test_input = test_input.to(device)
    reconstructed = tune.best_model(test_input)

    tune.plot_all_results(plots_suffix, is_accuracy=False, is_gridsearch=False, n_part=n_part)

    test_input = np.squeeze(test_input.cpu().detach().numpy(), 2)
    reconstructed = np.squeeze(reconstructed.cpu().detach().numpy(), 2)

    return test_input, reconstructed


def reconstruct():
    data_dir = os.path.join("data", "SP 500 Stock Prices 2014-2017.csv")
    plots_suffix = os.path.join("plots", "snp500", "part_II")
    config = {"hidden_size": 256,
              "lr": 0.001,
              "grad_clip": None}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_tensor, train_stocks_names = DataUtils.load_snp500(data_dir, args.batch_size, 10)
    test_input, reconstructed = snp500_reconstruct(data_tensor[0],
                                                   config,
                                                   device,
                                                   plots_suffix,
                                                   "_part"+str(0)+"_")
    for i in range(len(data_tensor) - 1):
        sub_test_input, sub_reconstructed = snp500_reconstruct(data_tensor[i + 1],
                                                               config,
                                                               device,
                                                               plots_suffix,
                                                               "_part"+str(i + 1)+"_")

        test_input = np.concatenate((test_input, sub_test_input), axis=1)
        reconstructed = np.concatenate((reconstructed, sub_reconstructed), axis=1)

    VisualizationUtils.plot_reconstruct(reconstructed,
                                        test_input,
                                        3,
                                        os.path.join(plots_suffix, "reconstruct"),
                                        "Reconstructed vs Original",
                                        ["Origin", "Reconstructed"])


def snp500_prediction(data_tensor, config, device, plots_suffix, n_part):
    train_idxs, test_idxs = DataUtils.create_random_train_test_indices_split(data_tensor.shape[0], 0.85, 0.15)
    data_gen = DataUtils.generate_random_split(data_tensor[train_idxs], args.folds, 0.8, 0.2)
    test_loader = DataUtils.create_data_loader(data_tensor[test_idxs].unsqueeze(2), args.batch_size)
    criterion = nn.MSELoss()
    tune = ParameterTuning()
    tune.kfold_run(train_func=partial(TrainingUtils.kfold_train,
                                      # auto_encoder_init=ToyAutoEncoder,
                                      auto_encoder_init=SP500AutoEncoder_prediction,
                                      lr=config["lr"],
                                      hidden_size=config["hidden_size"],
                                      input_size=2,
                                      input_seq_size=int(data_tensor.shape[1] / 2),  # 1007
                                      batch_size=args.batch_size,
                                      criterion=criterion,
                                      optimizer=args.optimizer,
                                      lstm_layers_size=args.lstm_layers_size,
                                      decoder_output_size=args.decoder_output_size,
                                      epochs=args.epochs,
                                      training_iteration=TrainingUtils.prediction_training_iteration,
                                      validation=TrainingUtils.prediction_validation,
                                      device=device),
                   test_func=partial(TrainingUtils.prediction_test,
                                     criterion=criterion,
                                     test_loader=test_loader,
                                     device=device),
                   data_tensor=data_tensor[train_idxs].unsqueeze(2),
                   data_generator=data_gen,
                   batch_size=args.batch_size,
                   config=config)

    test_loader = DataUtils.create_data_loader(data_tensor[test_idxs].unsqueeze(2),
                                               min(len(test_idxs), args.batch_size))

    test_input = next(iter(test_loader))
    test_input = test_input.to(device)
    b, r, c = test_input.shape
    test_input = test_input.view(b, int(r / 2), 2)
    reconstruct, predict = tune.best_model(test_input)

    tune.plot_all_results(plots_suffix, is_accuracy=False, is_gridsearch=False, n_part=n_part)

    original_seq_first_day = test_input[:, :, 0].cpu().detach().numpy()
    original_seq_second_day = test_input[:, :, 1].cpu().detach().numpy()
    reconstruct = reconstruct.detach().cpu().numpy()
    predict = predict.detach().cpu().numpy()

    return original_seq_first_day, reconstruct, original_seq_second_day, predict


def prediction():
    data_dir = os.path.join("data", "SP 500 Stock Prices 2014-2017.csv")
    plots_suffix = os.path.join("plots", "snp500", "part_III")
    config = {"hidden_size": 256,
              "lr": 0.001,
              "grad_clip": None}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_tensor, train_stocks_names = DataUtils.load_snp500_double_input(data_dir, args.batch_size, 10)
    seq_first_day, reconstruct, seq_second_day, predict = snp500_prediction(data_tensor[0],
                                                                            config,
                                                                            device,
                                                                            plots_suffix,
                                                                            "_part"+str(0)+"_")

    for i in range(len(data_tensor) - 1):
        sub_seq_first_day, sub_reconstruct, sub_seq_second_day, sub_predict = snp500_prediction(data_tensor[i + 1],
                                                                                                config,
                                                                                                device,
                                                                                                plots_suffix,
                                                                                                "_part"+str(i + 1)+"_")

        seq_first_day = np.concatenate((seq_first_day, sub_seq_first_day), axis=1)
        reconstruct = np.concatenate((reconstruct, sub_reconstruct), axis=1)
        seq_second_day = np.concatenate((seq_second_day, sub_seq_second_day), axis=1)
        predict = np.concatenate((predict, sub_predict), axis=1)

    VisualizationUtils.plot_reconstruct(reconstruct,
                                        seq_first_day,
                                        3,
                                        os.path.join(plots_suffix, "Reconstructed"),
                                        "Reconstructed vs Original",
                                        ["Origin", "Reconstructed"])

    VisualizationUtils.plot_reconstruct(predict,
                                        seq_second_day,
                                        3,
                                        os.path.join(plots_suffix, "predict"),
                                        "Prediction vs Original",
                                        ["Origin", "Prediction"])


def plot_stocks():
    path = os.path.join("data", "SP 500 Stock Prices 2014-2017.csv")
    plots_suffix = os.path.join("plots", "snp500", "part_I")
    plot_stock_high_prices(path, plots_suffix)


if __name__ == "__main__":
    plot_stocks()   #3.1
    reconstruct()   #3.2
    prediction()    #3.3
