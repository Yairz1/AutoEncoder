from typing import Dict, List
from itertools import product
from matplotlib import pyplot as plt
from torch import nn

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
    def __init__(self, config_options: Dict[str, List]):
        """
        :param config_options:
        """
        self._config_options: Dict[str, List] = config_options
        self._param_space: ParamSpace = ParamSpace(config_options)
        self._best_config: Dict = dict()
        self._best_loss: float = float("inf")
        self._best_model: nn.Module = nn.Module()
        self.config2train_info: Dict[str, List] = dict()
        self.config2val_info: Dict[str, List] = dict()

    def run(self, train_func, test_func):
        """
        :param test_func:
        :param train_func:
        :return:
        """
        for config in self._param_space:
            auto_encoder, train_info, val_info = train_func(config)
            self.config2train_info[str(config)] = train_info
            self.config2val_info[str(config)] = val_info
            test_loss = test_func(auto_encoder)
            if test_loss < self._best_loss:
                self._best_config = config
                self._best_loss = test_loss
                self._best_model = auto_encoder

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
        ParameterTuning._plot_dict(path, self.config2val_info, "Configuration and validation information")

    def plot_train_trails(self, path: str):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        ParameterTuning._plot_dict(path, self.config2train_info, "Configuration and training information")

    def plot_best_train(self, path):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        ParameterTuning._single_plot(ax, self.config2train_info[config_key], config_key)
        fig.suptitle("Best training info")
        fig.plot()
        if path:
            ax.save(path)

    def plot_best_val(self, path):
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        ParameterTuning._single_plot(ax, self.config2val_info[config_key], config_key)
        fig.suptitle("Best validation info")
        fig.plot()
        if path:
            ax.save(path)

    @staticmethod
    def _plot_dict(path: str, config2info: Dict, title: str):
        fig, axs = plt.subplots(len(config2info))
        for ax, (config_str, config_info) in zip(axs, config2info.items()):
            ParameterTuning._single_plot(ax, config_info, config_str)
        fig.tight_layout(pad=3.0)
        fig.suptitle(title)
        if path:
            plt.savefig(path)
        plt.show()

    @staticmethod
    def _single_plot(ax, config_info, config_str):
        ax.plot(config_info, label="Data")
        ax.set_title(f"Configuration {config_str}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
