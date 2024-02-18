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
DATA_PKL_FILENAME_FEATURES_OUT = "01_75ms_feature_extract_out.pkl"


# ------------------------------------------------------------ #
# def main():
#     pass

# creating a training and test set
df_train: pd.DataFrame = pd.read_pickle(
    Path(DATA_INTERIM_PATH, DATA_PKL_FILENAME_FEATURES_OUT)
)

df_train.info()

# dropping cols we do not need -- need to keep the ex col though
df_train = df_train.drop(["participant", "set", "effort"], axis=1)

# dropping labels for training set
X = df_train.drop("ex", axis=1)
# labels only
y = df_train["ex"]

# training/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ------------------------------------------------------------ #
# for now we'll comment out because of periodic
# runs in jupyter interactive
# if __name__ == "__main__":
#     main()
