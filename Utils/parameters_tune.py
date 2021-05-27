from typing import Dict, List, Tuple
from itertools import product

from torch import nn

"""
Use example:
import train

config_options = {"a": [1, 2, 3], "b": [4, 5, 6]}
param_tuning = ParameterTuning(config_options)
best_res, tune_info = param_tuning.run(train)
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
        self.config2info: Dict[str, Tuple[List, List]] = dict()

    def run(self, train_func, test_func):
        """
        :param test_func:
        :param train_func:
        :return:
        """
        for config in self._param_space:
            auto_encoder, self.config2info[str(config)] = train_func(config)
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
