import os

from Utils.data_utils import DataUtils
from Utils.visualization_utils import VisualizationUtils


def main():
    path = os.path.join("data", "SP 500 Stock Prices 2014-2017.csv")
    amazon_daily_max, googl_daily_max = DataUtils.load_snp500_amzn_google_daily_max(path)
    VisualizationUtils.plot_df_columns(amazon_daily_max,
                                       "date",
                                       "high",
                                       "Amazon \nTime vs Daily Maximum",
                                       "Time",
                                       "Daily high")
    VisualizationUtils.plot_df_columns(googl_daily_max,
                                       "date",
                                       "high",
                                       "Google \nTime vs Daily Maximum",
                                       "Time",
                                       "Daily high")


if __name__ == "__main__":
    main()
