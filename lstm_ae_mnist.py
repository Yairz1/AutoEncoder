from Architectures.lstm_autoencoder import MnistAutoEncoder, MnistAutoEncoderClassifier
from Utils.data_utils import DataUtils
from Utils.parameters_tune import ParameterTuning
from Utils.training_utils import TrainingUtils
from Utils.visualization_utils import VisualizationUtils

import torch
from torch import nn
import os
from functools import partial
import argparse

parser = argparse.ArgumentParser(description='lstm_ae_toy')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',  # 50
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lstm-layers-size', type=int, default=3, metavar='N',
                    help='lstm layers number, default 3')
parser.add_argument('--lstm-dropout', type=int, default=0.3, metavar='N',
                    help='lstm layers number, default 0')
parser.add_argument('--optimizer', type=str, default="adam", metavar='N',
                    help='optimizer, default adam')
parser.add_argument('--load', type=bool, default=True, metavar='N',
                    help='To load or create new data, default True')
parser.add_argument('--input-size', type=int, default=28, metavar='N',
                    help='LSTM feature input size, default 1')
parser.add_argument('--seq-len', type=int, default=28, metavar='N',
                    help='LSTM sequence series length, default 784')
parser.add_argument('--decoder-output-size', type=int, default=28, metavar='N',
                    help='LSTM sequence series length, default 784')
args = parser.parse_args()
print(torch.cuda.get_device_name(0))


def compare_mnist_reconstruction(device, test_loader, model, path):
    with torch.no_grad():
        test_input, _ = next(iter(test_loader))
        test_input = test_input.to(device)
        reconstructed = model(test_input)
        VisualizationUtils.plot_mnist_reconstruct(reconstructed.cpu(), test_input.cpu(), (3, 2), path,
                                                  "Left: reconstructed\n Right: original")


def compare_mnist_reconstruction_classification(device, test_loader, model, path):
    with torch.no_grad():
        test_input, labels = next(iter(test_loader))
        test_input = test_input.to(device)
        reconstructed, predictions = model(test_input)
        VisualizationUtils.plot_mnist_reconstruct_classification(reconstructed.cpu(),
                                                                 test_input.cpu(),
                                                                 predictions.cpu(),
                                                                 labels.cpu(),
                                                                 (3, 2),
                                                                 path,
                                                                 "Left: reconstructed\n Right: original")


def mnist_reconstructing():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # plots_suffix = os.path.join("plots", "job_plots")
    plots_suffix = os.path.join("plots", "mnist", "part_I")
    data_dir = os.path.join("data")  # 196
    config = {"hidden_size": [196],
              "lr": [0.001],
              "grad_clip": [None]}
    test_loader, train_loader, _ = DataUtils.data_factory("mnist", data_dir, args.batch_size, True)
    VisualizationUtils.plot_mnist(path=os.path.join(plots_suffix, "example"), n=3, loader=train_loader)
    criterion = nn.MSELoss()
    tune = ParameterTuning(config_options=config)
    tune.run(train_func=partial(TrainingUtils.train,
                                auto_encoder_init=MnistAutoEncoder,
                                input_size=args.input_size,
                                input_seq_size=args.seq_len,
                                dataset_name="mnist",
                                batch_size=args.batch_size,
                                criterion=criterion,
                                optimizer=args.optimizer,
                                lstm_layers_size=args.lstm_layers_size,
                                decoder_output_size=args.decoder_output_size,
                                epochs=args.epochs,
                                load_data=args.load,
                                device=device,
                                training_iteration=TrainingUtils.training_iteration,
                                validation=TrainingUtils.validation,
                                data_dir=data_dir),
             test_func=partial(TrainingUtils.test_accuracy,
                               criterion=criterion,
                               test_loader=test_loader,
                               device=device))

    # compare_mnist_reconstruction(device, test_loader, tune.best_model, os.path.join(plots_suffix, "reconstruct"))
    compare_mnist_reconstruction(device,
                                 test_loader,
                                 tune.best_model,
                                 os.path.join(plots_suffix,
                                              "reconstruct"))

    tune.plot_all_results(plots_suffix, is_accuracy=False, is_gridsearch=False)


def mnist_classifying():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # plots_suffix = os.path.join("plots", "job_plots")
    plots_suffix = os.path.join("plots", "mnist", "part_II")
    data_dir = os.path.join("data")  # 196
    config = {"hidden_size": [196],
              "lr": [0.001],
              "grad_clip": [None]}
    test_loader, train_loader, _ = DataUtils.data_factory("mnist", data_dir, args.batch_size, True)
    VisualizationUtils.plot_mnist(path=os.path.join(plots_suffix, "example"), n=3, loader=train_loader)
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    # criterion = lambda output, target: loss(output, target[0])

    tune = ParameterTuning(config_options=config)
    tune.run(train_func=partial(TrainingUtils.train,
                                auto_encoder_init=MnistAutoEncoderClassifier,
                                input_size=args.input_size,
                                input_seq_size=args.seq_len,
                                dataset_name="mnist",
                                batch_size=args.batch_size,
                                criterion=mse_criterion,
                                optimizer=args.optimizer,
                                lstm_layers_size=args.lstm_layers_size,
                                decoder_output_size=args.decoder_output_size,
                                epochs=args.epochs,
                                load_data=args.load,
                                device=device,
                                training_iteration=partial(TrainingUtils.classification_training_iteration,
                                                           ce_criterion=ce_criterion),
                                validation=partial(TrainingUtils.classification_validation,
                                                   ce_criterion=ce_criterion),
                                data_dir=data_dir),
             test_func=partial(partial(TrainingUtils.classification_test_accuracy,
                                       ce_criterion=ce_criterion),
                               criterion=mse_criterion,
                               test_loader=test_loader,
                               device=device),
             collect_accuracy_info=True)

    # compare_mnist_reconstruction(device, test_loader, tune.best_model, os.path.join(plots_suffix, "reconstruct"))
    compare_mnist_reconstruction_classification(device,
                                                test_loader,
                                                tune.best_model,
                                                os.path.join(plots_suffix,
                                                             "reconstruct and classifying"))

    tune.plot_all_results(plots_suffix, is_accuracy=True, is_gridsearch=False)


if __name__ == "__main__":
    mnist_reconstructing()
    mnist_classifying()
