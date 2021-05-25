from typing import Optional, Union
import os.path
import torch
from torch.utils.data import random_split
from torch import device


class DataUtils:
    @staticmethod
    def create_synthetic_data(size: int, sample_size: int,
                              device_type: Union[str, device],
                              path: str = Optional) -> torch.Tensor:
        """
        Create a synthetic dataset or load from the given path
        :param device_type: gpu or cpu (cuda:0 or cpu)
        :param size: number of samples
        :param sample_size: sample size
        :param path: Optional parameter, a path to save and load the dataset from if exists
        :return: Torch tensor
        """
        if path and os.path.isfile(path):
            return torch.load(path)
        data = torch.rand(size=(size, sample_size), device=device_type)
        torch.save(data, path)
        return data

    @staticmethod
    def train_val_test_split(data: torch.Tensor, train_ratio: float, val_ratio: float, test_ratio: float):
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

