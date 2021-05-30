from typing import Dict, List, Union
from itertools import product
from matplotlib import pyplot as plt
from torch import nn

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
    def __init__(self, config_options: Dict[str, List], collect_accuracy_info: bool):
        """
        :param config_options:
        """
        self._config_options: Dict[str, List] = config_options
        self._param_space: ParamSpace = ParamSpace(config_options)
        self._best_config: Dict = dict()
        self._best_loss: float = float("inf")
        self._best_loss2: float = float("inf")
        self._best_accuracy: float = float("inf")
        self._best_model: Union[nn.Module, None] = None
        self.config2train_info: Dict[str, List] = dict()
        self.config2train_info2: Dict[str, List] = dict()
        self.config2val_info: Dict[str, List] = dict()
        self.config2val_info2: Dict[str, List] = dict()
        self.config2accuracy_train_info: Dict[str, List] = dict()
        self.config2accuracy_val_info: Dict[str, List] = dict()
        self.collect_accuracy_info: collect_accuracy_info = collect_accuracy_info

    def run(self, train_func, test_func):
        """
        :param test_func:
        :param train_func:
        :return:
        """
        for config in self._param_space:
            print(f"Running config: {config}")
            if self.collect_accuracy_info:
                auto_encoder, train_loss, train_loss2, val_loss, val_loss2, accuracy_train, accuracy_val = train_func(config)
                self.config2accuracy_train_info[str(config)] = accuracy_train
                self.config2accuracy_val_info[str(config)] = accuracy_val
                self.config2train_info2[str(config)] = train_loss2
                self.config2val_info2[str(config)] = val_loss2
            else:
                auto_encoder, train_loss, _, val_loss, _, _, _ = train_func(config)

            self.config2train_info[str(config)] = train_loss
            self.config2val_info[str(config)] = val_loss

            test_loss, test_loss2, test_accuracy = test_func(auto_encoder)
            if test_loss < self._best_loss:
                self._best_config = config
                self._best_loss = test_loss
                self._best_loss2 = test_loss2
                self._best_model = auto_encoder
                self._best_accuracy = test_accuracy

    @property
    def best_config(self):
        return self._best_config

    @property
    def best_loss(self):
        return self._best_loss

    @property
    def best_loss2(self):
        return self._best_loss2

    @property
    def best_model(self):
        return self._best_model

    @property
    def best_accuracy(self):
        return self._best_accuracy

    def get_best_val_loss(self):
        if not self.config2val_info:
            raise Exception("Execute run method first")
        return min(self.config2val_info[str(self._best_config)])

    def get_best_val_loss2(self):
        if not self.config2val_info:
            raise Exception("Execute run method first")
        return min(self.config2val_info2[str(self._best_config)])

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

    def plot_best_train_mse(self, path):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        VisualizationUtils.single_plot(ax, self.config2train_info[config_key], config_key)
        fig.suptitle("Best training info mse loss")
        fig.show()
        if path:
            fig.savefig(path)

    def plot_best_train_ce(self, path):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        VisualizationUtils.single_plot(ax, self.config2train_info2[config_key], config_key)
        fig.suptitle("Best training info ce loss")
        fig.show()
        if path:
            fig.savefig(path)

    def plot_best_val_mse(self, path):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        VisualizationUtils.single_plot(ax, self.config2val_info[config_key], config_key)
        fig.suptitle("Best val info mse loss")
        fig.show()
        if path:
            fig.savefig(path)

    def plot_best_val_ce(self, path):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        VisualizationUtils.single_plot(ax, self.config2val_info2[config_key], config_key)
        fig.suptitle("Best val info ce loss")
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

    def plot_best_accuracy_val(self, path):
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        VisualizationUtils.classification_single_plot(ax, self.config2accuracy_val_info[config_key], config_key)
        fig.suptitle("Best accuracy validation info")
        fig.show()
        if path:
            fig.savefig(path)

    def plot_best_accuracy_train(self, path):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        VisualizationUtils.classification_single_plot(ax, self.config2accuracy_train_info[config_key], config_key)
        fig.suptitle("Best accuracy training info")
        fig.show()
        if path:
            fig.savefig(path)