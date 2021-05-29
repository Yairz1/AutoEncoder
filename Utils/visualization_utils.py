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
            ax.set_ylim((0, 1))

        handles, labels = axs[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        fig.tight_layout(pad=3.0)
        fig.suptitle(title)
        plt.show()
        if path:
            fig.savefig(path)

    @staticmethod
    def visualize_data_reconstruction(reconstruction: torch.tensor,
                                      data: torch.tensor,
                                      n: int,
                                      title: str,
                                      xlabel: str,
                                      ylabel: str,
                                      path: str):
        """

        :param reconstruction:
        :param data:
        :param n:
        :param title:
        :param xlabel:
        :param ylabel:
        :param path:
        :return:
        """

        fig, axs = plt.subplots(n)
        for i, ax in enumerate(axs):
            ax.plot(data[i, :], label="Data")
            ax.plot(reconstruction[i, :], label="Data")
            ax.set_title(f"Sample {i}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        handles, labels = axs[-1].get_legend_handles_labels()
        fig.legend(handles, ["Origin", "Reconstructed"], loc='upper left')
        fig.tight_layout(pad=3.0)
        fig.suptitle(title)
        plt.show()
        if path:
            fig.savefig(path)

    @staticmethod
    def plot_dict(path: str, config2info: Dict, title: str):
        fig, axs = plt.subplots(len(config2info))
        if len(config2info) > 1:
            for ax, (config_str, config_info) in zip(axs, config2info.items()):
                VisualizationUtils.single_plot(ax, config_info, config_str)
            fig.tight_layout(pad=5.0)
        else:
            VisualizationUtils.single_plot(axs, list(config2info.values())[0], list(config2info.keys())[0])
        fig.suptitle(title)
        plt.show()
        if path:
            fig.savefig(path)

    @staticmethod
    def single_plot(ax, config_info, config_str):
        ax.plot(config_info, label="Data")
        ax.set_title(f"Configuration {config_str}", pad=3)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")

    @staticmethod
    def plot_reconstruct(reconstruction: torch.tensor, test_input: torch.tensor, n: int, path: str):
        """

        :param reconstruction:
        :param test_input:
        :param n:
        :return:
        """

        VisualizationUtils.visualize_data_reconstruction(reconstruction=reconstruction,
                                                         data=test_input,
                                                         n=n,
                                                         title="Reconstructed vs Original",
                                                         xlabel="Time",
                                                         ylabel="Value",
                                                         path=path)
