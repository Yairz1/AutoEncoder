from typing import Union, Optional, Type, Any, Tuple

import os
import torch
from torch.utils.data import random_split, DataLoader
from torch import device
from torch.distributions.uniform import Uniform
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import random_split


class DataUtils:
    @staticmethod
    def data_loader_factory(dataset_name: str, path: str, batch_size: int, load: bool) -> Tuple:
        if dataset_name.lower() == "mnist":
            return DataUtils.load_mnist(root=path, batch_size=batch_size)
        elif dataset_name.lower() == "synthetic_data":
            return DataUtils.load_synthetic_data(path, batch_size, load)
        else:
            raise Exception("Dataset not supported")

    class Normalization:
        def __init__(self, average: float):
            self.average = average

        def __call__(self, sample):
            _min = sample.min()
            _max = sample.max()
            sample = (sample - _min) / (2 * (_max - _min))
            sample = sample - sample.mean() + 0.5
            return sample

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
    def train_val_test_split(data: torch.tensor, train_ratio: float, val_ratio: float, test_ratio: float):
        """
        return splitting data if not exist in path
        :param data: data to split
        :param train_ratio: in (0,1)
        :param val_ratio:  in (0,1)
        :param test_ratio:  in (0,1)
        :return: train, val and test sets
        """
        if train_ratio + val_ratio + test_ratio != 1 and not (
                0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
            raise Exception(f"Train - val - test ratio are not valid {train_ratio}|{val_ratio}|{test_ratio}")
        data_size = data.shape[0]
        train_size = int(data_size * train_ratio)
        val_size = int(data_size * val_ratio)
        test_size = int(data_size * test_ratio)
        return random_split(dataset=data, lengths=(train_size, val_size, test_size))

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
    def load_mnist(root, batch_size=128):
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
