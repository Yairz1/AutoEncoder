from typing import Dict, List, Union
from itertools import product
from matplotlib import pyplot as plt
from torch import nn

from Utils.visualization_utils import VisualizationUtils

import os
import numpy as np

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
        self._test_loss: List[float]
        self._best_accuracy: float = float("inf")
        self._best_model: Union[nn.Module, None] = None
        self.config2train_info: Dict[str, dict] = dict()
        self.config2train_info2: Dict[str, List] = dict()
        self.config2val_info: Dict[str, dict] = dict()
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
            auto_encoder, train_info_dic, val_info_dic = train_func(config)

            if self.collect_accuracy_info:
                self.config2accuracy_train_info[str(config)] = train_info_dic.pop("accuracy", None)
                self.config2accuracy_val_info[str(config)] = val_info_dic.pop("accuracy", None)

            self.config2train_info[str(config)] = train_info_dic
            self.config2val_info[str(config)] = val_info_dic

            test_info = test_func(auto_encoder)
            test_accuracy = test_info.pop("accuracy", None)
            test_total_loss = sum(list(test_info.values()))

            if test_total_loss < self._best_loss:
                self._best_config = config
                self._best_loss = test_total_loss
                self._test_loss = list(test_info.values())
                self._best_model = auto_encoder
                if self.collect_accuracy_info:
                    self._best_accuracy = test_accuracy

    @property
    def best_config(self):
        return self._best_config

    @property
    def best_loss(self):
        return np.array(self._test_loss)

    # @property
    # def best_loss2(self):
    #     return self._test_loss

    @property
    def best_model(self):
        return self._best_model

    @property
    def best_accuracy(self):
        return self._best_accuracy

    def get_best_val_loss(self):
        if not self.config2val_info:
            raise Exception("Execute run method first")
        values = np.array(list(self.config2val_info[str(self._best_config)].values()))
        return np.amin(values, 1)

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

    def plot_accuracy(self, config_info, path, title, xlabel, ylabel):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        path = os.path.join(path, title)
        config_key = str(self.best_config)
        fig, ax = plt.subplots()
        VisualizationUtils.single_plot(ax, config_info[config_key], config_key, xlabel, ylabel)
        fig.suptitle(title)
        fig.show()
        if path:
            fig.savefig(path)

    def plot_best_loss(self, config_info, sub_path, title, xlabel, ylabel):
        if not self.config2train_info:
            raise Exception("Execute run method first")
        config_key = str(self.best_config)
        info_dic = config_info[config_key]
        for key, value in info_dic.items():
            fig, ax = plt.subplots()
            path = os.path.join(sub_path, title+"_"+key)
            VisualizationUtils.single_plot(ax, value, config_key, xlabel, ylabel)
            fig.suptitle(title+"-"+key)
            fig.show()
            if path:
                fig.savefig(path)

    def plot_all_results(self, plots_suffix, is_accuracy, is_gridsearch):

        print("Best trial config: {}".format(self.best_config))
        print("Best trial validation loss: {}".format(np.round(self.get_best_val_loss(), 3)))
        print("Best trial test total loss: {}".format(np.round(self.best_loss, 3)))
        self.plot_best_loss(self.config2train_info, plots_suffix, "best_train_trail_loss", "Epochs", "Loss")
        self.plot_best_loss(self.config2val_info, plots_suffix, "best_validation_trail_loss", "Epochs", "Loss")

        if is_gridsearch:
            self.plot_validation_trails(path=os.path.join(plots_suffix, "all_validation_trails"))
            self.plot_train_trails(path=os.path.join(plots_suffix, "all_train_trails"))

        if is_accuracy:
            print("Best accuracy of the network on test set : {}".format(round(self.best_accuracy, 3)))
            self.plot_accuracy(self.config2accuracy_train_info, plots_suffix, "best_accuracy_train_trail",
                                       "Accuracy", "Loss")
            self.plot_accuracy(self.config2accuracy_val_info, plots_suffix, "best_accuracy_validation_trail",
                                       "Accuracy", "Loss")


    # def plot_best_train(self, path):
    #     if not self.config2train_info:
    #         raise Exception("Execute run method first")
    #     config_key = str(self.best_config)
    #     fig, ax = plt.subplots()
    #     VisualizationUtils.single_plot(ax, self.config2train_info[config_key], config_key)
    #     fig.suptitle("Best training info")
    #     fig.show()
    #     if path:
    #         fig.savefig(path)
    #
    # def plot_best_val(self, path):
    #     config_key = str(self.best_config)
    #     fig, ax = plt.subplots()
    #     VisualizationUtils.single_plot(ax, self.config2val_info[config_key], config_key)
    #     fig.suptitle("Best validation info")
    #     fig.show()
    #     if path:
    #         fig.savefig(path)
    #
    # def plot_best_accuracy_val(self, path):
    #     config_key = str(self.best_config)
    #     fig, ax = plt.subplots()
    #     VisualizationUtils.classification_single_plot(ax, self.config2accuracy_val_info[config_key], config_key)
    #     fig.suptitle("Best accuracy validation info")
    #     fig.show()
    #     if path:
    #         fig.savefig(path)
    #
    # def plot_best_accuracy_train(self, path):
    #     if not self.config2train_info:
    #         raise Exception("Execute run method first")
    #     config_key = str(self.best_config)
    #     fig, ax = plt.subplots()
    #     VisualizationUtils.classification_single_plot(ax, self.config2accuracy_train_info[config_key], config_key)
    #     fig.suptitle("Best accuracy training info")
    #     fig.show()
    #     if path:
    #         fig.savefig(path)
