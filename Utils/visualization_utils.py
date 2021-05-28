from typing import Dict

import torch
from matplotlib import pyplot as plt


class VisualizationUtils:
    @staticmethod
    def visualize_data_examples(data: torch.tensor, n: int, title: str, xlabel: str, ylabel: str, path: str):
        """
        :param data: Data to visualize
        :param n: number of subplots
        :param title: figure title
        :param xlabel: x label
        :param ylabel: y label
        :param path: path to save the figure
        :return: Presents a plot
        """
        fig, axs = plt.subplots(n)
        for i, ax in enumerate(axs):
            ax.plot(data[i, :], label="Data")
            ax.set_title(f"Sample {i}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axhline(0.5, color='red', ls='--', label='Mean')

        handles, labels = axs[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        fig.tight_layout(pad=3.0)
        fig.suptitle(title)
        if path:
            plt.savefig(path)
        plt.show()

    @staticmethod
    def plot_dict(path: str, config2info: Dict, title: str):
        fig, axs = plt.subplots(len(config2info))
        for ax, (config_str, config_info) in zip(axs, config2info.items()):
            VisualizationUtils.single_plot(ax, config_info, config_str)
        fig.tight_layout(pad=5.0)
        fig.suptitle(title)
        if path:
            plt.savefig(path)
        plt.show()

    @staticmethod
    def single_plot(ax, config_info, config_str):
        ax.plot(config_info, label="Data")
        ax.set_title(f"Configuration {config_str}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")

    @staticmethod
    def plot_context(context: torch.tensor, test_input: torch.tensor, n: int):
        """

        :param context:
        :param test_input:
        :param n:
        :return:
        """
        VisualizationUtils.visualize_data_examples(context, 3, "Reconstructed vector", "Time", "Value", "")
        VisualizationUtils.visualize_data_examples(test_input, 3, "Original vector", "Time", "Value", "")
