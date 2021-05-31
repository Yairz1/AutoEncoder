from typing import Dict, Tuple

import pandas as pd
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
        if len(config2info) > 1:
            fig, axs = plt.subplots(len(config2info), figsize=(10, 30))
            for ax, (config_str, config_info) in zip(axs, config2info.items()):
                VisualizationUtils.single_plot(ax, list(config_info.values())[0], config_str, "Epochs", "Loss")
                fig.tight_layout(pad=5.0)
        else:
            fig, axs = plt.subplots(len(config2info), figsize=(10, 30))
            VisualizationUtils.single_plot(axs,
                                           list(config2info.values())[0],
                                           list(config2info.keys())[0],
                                           "Epochs",
                                           "Loss")
        fig.suptitle(title)
        plt.show()
        if path:
            fig.savefig(path)

    @staticmethod
    def single_plot(ax, config_info, config_str, xlabel, ylabel):
        ax.plot(list(config_info), label="Data")
        ax.set_title(f"Configuration {config_str}", pad=3)
        ax.set_xlabel(xlabel)  # "Epochs"
        ax.set_ylabel(ylabel)   #"Loss

    @staticmethod
    def classification_single_plot(ax, config_info, config_str):
        ax.plot(config_info, label="Data")
        ax.set_title(f"Configuration {config_str}", pad=3)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")

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

    @staticmethod
    def plot_mnist_reconstruct(reconstruction: torch.tensor,
                               test_input: torch.tensor,
                               n: Tuple[int, int],
                               path: str,
                               title: str):
        """
        :param reconstruction:
        :param test_input:
        :param n:
        :param path:
        :param title:
        :return:
        """

        fig, axs = plt.subplots(n[0], n[1])
        for i in range(n[0]):
            axs[i, 0].imshow(reconstruction[i], cmap="gray")
            axs[i, 1].imshow(test_input[i], cmap="gray")

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, ["Origin", "Reconstructed"], loc='upper left')
        fig.tight_layout(pad=3.0)
        fig.suptitle(title)
        plt.show()
        if path:
            fig.savefig(path)

    @staticmethod
    def plot_mnist_reconstruct_classification(reconstruction: torch.tensor,
                               test_input: torch.tensor,
                               predictions: torch.tensor,
                               labels: torch.tensor,
                               n: Tuple[int, int],
                               path: str,
                               title: str):
        """
        :param reconstruction:
        :param test_input:
        :param predictions:
        :param labels:
        :param n:
        :param path:
        :param title:
        :return:
        """

        fig, axs = plt.subplots(n[0], n[1])
        for i in range(n[0]):
            axs[i, 0].imshow(reconstruction[i], cmap="gray")
            axs[i, 1].imshow(test_input[i], cmap="gray")
            axs[i, 0].set_title(f"Predicted Digit {torch.argmax(predictions[i])}")
            axs[i, 1].set_title(f"Digit {labels[i]}")

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, ["Origin", "Reconstructed"], loc='upper left')
        fig.tight_layout(pad=3.0)
        fig.suptitle(title)
        plt.show()
        if path:
            fig.savefig(path)

    @staticmethod
    def plot_mnist(path, n, loader):
        images, labels = next(iter(loader))
        fig, axs = plt.subplots(n)
        for i, ax in enumerate(axs):
            ax.imshow(images[i], cmap="gray")
            ax.set_title(f"Digit {labels[i]}")
        fig.tight_layout(pad=0.5)
        fig.show()
        if path:
            fig.savefig(path)

    @staticmethod
    def plot_df_columns(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str):
        df.plot(x=x_col, y=y_col, title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()
