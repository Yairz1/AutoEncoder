from typing import Dict, List, Union
from itertools import product
from matplotlib import pyplot as plt
from torch import nn

from Utils.data_utils import DataUtils
from Utils.visualization_utils import VisualizationUtils

"""
Use example:
import train

config_options = {"a": [1, 2, 3], "b": [4, 5, 6]}
param_tuning = ParameterTuning(config_options)
tune.run(train)
"""


class ParamSpace:
    def __init__(self, config_options: Dict[str, List]):
        self.config_keys = list(config_options.keys())
        self.param_space = product(*config_options.values())

    def __iter__(self):
        for point in self.param_space:
            yield dict(zip(self.config_keys, point))


class ParameterTuning:
    def __init__(self, config_options: Dict[str, List] = None):
        """
        :param config_options:
        """
        self._config_options: Dict[str, List] = config_options
        if config_options:
            self._param_space: ParamSpace = ParamSpace(config_options)
        self._best_config: Dict = dict()
        self._best_val_loss: float = float("inf")
        self._test_loss: float = float("inf")
        self._best_fold: int = 0
        self._best_model: Union[nn.Module, None] = None
        self.config2train_info: Dict[str, List] = dict()
        self.config2val_info: Dict[str, List] = dict()
        self.fold_train_info: List[float] = []
        self.fold_val_info: List[float] = []

    def run(self, train_func, test_func):
        """
        :param test_func:
        :param train_func:
        :return:
        """
        for config in self._param_space:
            print(f"Running config: {config}")
            auto_encoder, train_info, val_info = train_func(config)
            self.config2train_info[str(config)] = train_info
            self.config2val_info[str(config)] = val_info
            test_loss = test_func(auto_encoder)
            if test_loss < self._best_loss:
                self._best_config = config
                self._best_loss = test_loss
                self._best_model = auto_encoder

    def kfold_run(self, train_func, test_func, data_tensor, data_generator, batch_size):
        """

        :param train_func:
        :param test_func:
        :param data_tensor:
        :param data_generator:
        :param batch_size:
        :return:
        """
        for i, (tr_ind, val_ind) in enumerate(data_generator):
            print(f"K = {i}")
            train_loader = DataUtils.create_data_loader(data_tensor[tr_ind, :], batch_size)
            val_loader = DataUtils.create_data_loader(data_tensor[val_ind, :], batch_size)
            auto_encoder, train_info, val_info = train_func(train_loader, val_loader)
            self.fold_train_info.append(train_info)
            self.fold_val_info.append(val_info)
            if val_info[-1] < self._best_val_loss:
                self._best_fold = i
                self._best_model = auto_encoder

        self._test_loss = test_func(auto_encoder)

    @property
    def best_config(self):
        return self._best_config

    @property
    def best_loss(self):
        return self._best_loss

    @property
    def best_model(self):
        return self._best_model

    def get_best_val_loss(self):
        if not self.config2val_info:
            raise Exception("Execute run method first")
        return min(self.config2val_info[str(self._best_config)])

    def plot_validation_trails(self, path: str):
        if not self.config2val_info:
            raise Exception("Execute run method first")
        VisualizationUtils.plot_dict(path, self.config2val_info, "Configuration and validation information")

    def plot_train_trails(self, path: str):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        VisualizationUtils.plot_dict(path, self.config2train_info, "Configuration and training information")

    def plot_best_train(self, path):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        VisualizationUtils.single_plot(ax, self.config2train_info[config_key], config_key)
        fig.suptitle("Best training info")
        fig.show()
        if path:
            fig.savefig(path)

    def plot_best_val(self, path):
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        VisualizationUtils.single_plot(ax, self.config2val_info[config_key], config_key)
        fig.suptitle("Best validation info")
        fig.show()
        if path:
            fig.savefig(path)
