from Utils.data_utils import DataUtils
from Utils.visualization_utils import VisualizationUtils


def plot_synthetic_samples():
    synthetic_data = DataUtils.create_synthetic_data(size=100000,
                                                     sample_size=50,
                                                     device="cuda:0",
                                                     path="./data/synthetic_data").cpu()
    VisualizationUtils.visualize_data_examples(synthetic_data,
                                               n=3,
                                               title='Synthetic samples',
                                               xlabel='Time',
                                               ylabel='Value',
                                               path="./plots/synthetic_samples")

plot_synthetic_samples()