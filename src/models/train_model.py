import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ClassificationAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix


# ------------------------------------------------------------ #
# rcParams
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# ------------------------------------------------------------ #
DATA_PATH = Path("../../data")
DATA_INTERIM_PATH = Path(DATA_PATH, "interim")
DATA_PKL_FILENAME_OUTLIERS_REM = "01_75ms_outliers_removed_chauvenet.pkl"


# ------------------------------------------------------------ #
# def main():
#     pass

# TODO

# ------------------------------------------------------------ #
# for now we'll comment out because of periodic
# runs in jupyter interactive
# if __name__ == "__main__":
#     main()
