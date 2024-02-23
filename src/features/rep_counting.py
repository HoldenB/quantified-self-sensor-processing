import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

# ------------------------------------------------------------ #
# pd options
pd.options.mode.chained_assignment = None


# ------------------------------------------------------------ #
# rcParams
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# ------------------------------------------------------------ #
DATA_PATH = Path("../../data")
DATA_INTERIM_PATH = Path(DATA_PATH, "interim")
# starting with the simple processed data -- we lose too much
# information if we use the feature_extracted data
DATA_PKL_FILENAME = "01_75ms_data_processed.pkl"


# ------------------------------------------------------------ #
# def main():
#     pass

df: pd.DataFrame = pd.read_pickle(Path(DATA_INTERIM_PATH, DATA_PKL_FILENAME))

# ------------------------------------------------------------ #
# for now we'll comment out because of periodic
# runs in jupyter interactive
# if __name__ == "__main__":
#     main()
