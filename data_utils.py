import torch
from torch.utils.data import random_split


def create_synthetic_data(size: int, sample_size: int, device: str) -> torch.Tensor:
    """
    :param device: gpu or cpu (cuda:0 or cpu)
    :param size: number of samples
    :param sample_size: sample size
    :return: Torch tensor
    """
    return torch.rand(size=(size, sample_size), device=device)


def train_val_test_split(data, train_ratio: float, val_ratio: float, test_ratio: float):
    """
    return splitting data
    :param data: data to split
    :param train_ratio: in (0,1)
    :param val_ratio:  in (0,1)
    :param test_ratio:  in (0,1)
    :return: Splitting data
    """
    if train_ratio + val_ratio + test_ratio != 1 and not (
            0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        raise Exception(f"Train - val - test ratio are not valid {train_ratio}|{val_ratio}|{test_ratio}")
    data_size = data.shape[0]
    train_size = int(data_size * train_ratio)
    val_size = int(data_size * val_ratio)
    test_size = int(data_size * test_ratio)
    return random_split(dataset=data, lengths=(train_size, val_size, test_size))


X = create_synthetic_data(size=100000, sample_size=50, device="cuda:0")
