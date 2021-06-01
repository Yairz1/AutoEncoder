from typing import Any, Tuple

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import device
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import random_split
import pandas as pd


class DataUtils:
    @staticmethod
    def data_factory(dataset_name: str, path: str, batch_size: int, load: bool) -> Tuple:
        if dataset_name.lower() == "mnist":
            return DataUtils.load_mnist(root=path, batch_size=batch_size)
        elif dataset_name.lower() == "synthetic_data":
            return DataUtils.load_synthetic_data(path, batch_size, load)
        elif dataset_name.lower() == "sp500":
            return DataUtils.load_snp500(path, batch_size)
        else:
            raise Exception("Dataset not supported")

    class ReshapeTransform:
        def __init__(self, shape):
            self.shape = shape

        def __call__(self, x):
            return x.reshape(self.shape)

    @staticmethod
    def create_synthetic_data(size: int,
                              sample_size: int,
                              device_type: Any,
                              path: str,
                              load: bool = True) -> torch.tensor:
        """
        Create a synthetic dataset or load from the given path
        :param device_type: gpu or cpu (cuda:0 or cpu)
        :param size: number of samples
        :param sample_size: sample size
        :param path: path to save and load the dataset
        :param load: Either to load or create new data
        :return: Torch tensor
        """

        if load and path and os.path.isfile(path):
            return torch.load(path)
        data = torch.FloatTensor(size, sample_size).uniform_(0, 0.7)  # empirically the max value will be less then 1
        data = data - data.mean(1)[:, None] + 0.5
        # if path:
        #     dir_path = os.path.dirname(path)
        #     torch.save(data, dir_path)
        return data

    @staticmethod
    def train_val_test_split(data: torch.tensor, train_ratio: float, val_ratio: float, test_ratio: float,
                             with_test=True):
        """
        return splitting data if not exist in path
        :param data: data to split
        :param train_ratio: in (0,1)
        :param val_ratio:  in (0,1)
        :param test_ratio:  in (0,1)
        :return: train, val and test sets
        """
        if train_ratio + val_ratio + test_ratio != 1 and not (
                0 <= train_ratio < 1 and 0 <= val_ratio < 1 and 0 <= test_ratio < 1):
            raise Exception(f"Train - val - test ratio are not valid {train_ratio}|{val_ratio}|{test_ratio}")
        data_size = data.shape[0]
        if with_test:
            train_size = int(data_size * train_ratio)
            val_size = int(data_size * val_ratio)
            test_size = data_size - train_size - val_size
            return random_split(dataset=data, lengths=(train_size, val_size, test_size))
        train_size = int(data_size * train_ratio)
        val_size = data_size - train_size
        return random_split(dataset=data, lengths=(train_size, val_size))

    @staticmethod
    def load_synthetic_data(path, batch_size, load):
        data_size = 10000
        series_size = 50
        dataset = DataUtils.create_synthetic_data(data_size, series_size, device, path, load)
        dataset = dataset.unsqueeze(2)
        train, val, test = DataUtils.train_val_test_split(dataset, 0.6, 0.2, 0.2)
        train_loader = DataLoader(train, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test, batch_size=batch_size, drop_last=True)
        return test_loader, train_loader, val_loader

    @staticmethod
    def load_mnist(root, batch_size):
        """Reference https://github.com/pytorch/examples/blob/master/mnist/main.py"""
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        DataUtils.ReshapeTransform(shape=(28, 28))
                                        ])
        train_set = MNIST(root, train=True, download=True, transform=transform)
        test_set = MNIST(root, train=False, transform=transform)
        train_set, val_set = random_split(train_set, [int(len(train_set) * 0.8), int(len(train_set) * 0.2)])
        train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, drop_last=True)
        return train_loader, val_loader, test_loader

    @staticmethod
    def load_snp500_amzn_google_daily_max(path):
        s_p_500 = pd.read_csv(path)
        G_A = s_p_500[s_p_500['symbol'].isin(["AMZN", "GOOGL"])]
        G_A = G_A.sort_values(by="date")
        amazon_daily_max = G_A[G_A.symbol == "AMZN"]
        googl_daily_max = G_A[G_A.symbol == "GOOGL"]
        return amazon_daily_max, googl_daily_max

    @staticmethod
    def normalize(data):
        data -= data.min(1, keepdim=True)[0]
        data /= data.max(1, keepdim=True)[0]
        return data

    @staticmethod
    def generate_random_split(dataset, n, train_ratio, val_ratio):
        for _ in range(n):
            train, val = DataUtils.train_val_test_split(dataset, train_ratio, val_ratio, 0, with_test=False)
            yield train.indices, val.indices

    @staticmethod
    def create_random_train_test_indices_split(n, train, test):
        indices = np.random.permutation(n)
        train_size = int(n * train)
        return indices[:train_size], indices[train_size:]

    @staticmethod
    def load_snp500(path, batch_size, n_division):
        sp_500_df = pd.read_csv(path)  # , nrows=1000)
        sp_500_df = sp_500_df.sort_values(by="date")
        sp_500_df = sp_500_df[["symbol", "close"]]
        sp_500_group = sp_500_df.groupby('symbol')
        stocks_names = list(sp_500_group.groups.keys())
        sp500_array = sp_500_group['close'].apply(lambda x: pd.Series(x.values)).unstack()
        sp500_array.interpolate(inplace=True)
        sp500_tensor = DataUtils.normalize(torch.FloatTensor(sp500_array.values))

        sp500_tensor = np.array_split(torch.transpose(sp500_tensor, 1, 0), n_division)
        sp500_tensor = [torch.transpose(sp500_tensor[i], 1, 0) for i in range(n_division)]

        return sp500_tensor, stocks_names

    @staticmethod
    def load_snp500_double_input(path, batch_size, n_division):
        sp_500_df = pd.read_csv(path)  # , nrows=1000)
        sp_500_df = sp_500_df.sort_values(by="date")
        sp_500_df = sp_500_df[["symbol", "close"]]
        sp_500_group = sp_500_df.groupby('symbol')
        stocks_names = list(sp_500_group.groups.keys())
        sp500_array = sp_500_group['close'].apply(lambda x: pd.Series(x.values)).unstack()
        sp500_array.interpolate(inplace=True)

        double_input = np.concatenate((np.array(sp500_array)[:, 0:1006], np.array(sp500_array)[:, 1:1007]), axis=1)
        double_input = np.array(DataUtils.normalize(torch.FloatTensor(double_input)))

        first_input = torch.FloatTensor(double_input[:, 0:1006])
        first_input = np.array_split(torch.transpose(first_input, 1, 0), n_division)
        first_input = [torch.transpose(first_input[i], 1, 0) for i in range(n_division)]

        second_input = torch.FloatTensor(double_input[:, 1006:])
        second_input = np.array_split(torch.transpose(second_input, 1, 0), n_division)
        second_input = [torch.transpose(second_input[i], 1, 0) for i in range(n_division)]

        sp500_tensor = [torch.cat((first_input[i], second_input[i]), 1) for i in range(n_division)]

        return sp500_tensor, stocks_names

    @staticmethod
    def create_data_loader(data: torch.tensor, batch_size):
        return DataLoader(data, batch_size, drop_last=True)


