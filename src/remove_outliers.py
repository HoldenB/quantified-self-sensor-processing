import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path


# ------------------------------------------------------------ #
# rcParams
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100


# ------------------------------------------------------------ #
DATA_PATH = Path("../data")
DATA_INTERIM_PATH = Path(DATA_PATH, "interim")
DATA_PKL_FILENAME = "01_75ms_data_processed.pkl"


# ------------------------------------------------------------ #
def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typically between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0
            - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


# ------------------------------------------------------------ #
# def main():
#     pass

# note: we may eventually want to keep the outliers in the actual
# model to catch subtle movement patterns and variations between
# specific users

# reorder cols to aggregate accel & gyro indexes
data_processed_df: pd.DataFrame = pd.read_pickle(
    Path(DATA_INTERIM_PATH, DATA_PKL_FILENAME)
).reindex(
    columns=[
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "set",
        "participant",
        "ex",
        "effort",
    ]
)

# only the sensor data
outlier_cols = list(data_processed_df.columns[:6])

accel_cols = outlier_cols[:3]
gyro_cols = outlier_cols[3:]

data_processed_df[accel_cols + ["ex"]].plot.boxplot(
    by="ex", figsize=(20, 10), layout=(1, 3)
)

data_processed_df[gyro_cols + ["ex"]].plot.boxplot(
    by="ex", figsize=(20, 10), layout=(1, 3)
)
# plt.show()

# need to visualize outliers over time to determine normal values
# but we need to first mark outliers using IQR
# marking IQR will add additional cols marked for example: accel_x_outlier
for col in outlier_cols:
    outlier_marked_df = mark_outliers_iqr(data_processed_df, col)
    plot_binary_outliers(
        outlier_marked_df, col, f"{col}_outlier", reset_index=True
    )

# need to differentiate between exercises otherwise IQR will catch cross-
# set values that are under-represented and treat them as outliers
# also to note: rest periods the user could do whatever they liked:
# walk around, drink water, etc.

# need to group data by exercise -> look for outliers
# we can use Chauvenet’s Criterion to look for outliers
#
# according to Chauvenet’s criterion we reject a measurement
# (outlier) from a dataset of size N when it’s probability of
# observation is less than 1/2N. A generalization is to replace
# the value 2 with a parameter C
# note: we sort of make the assumption here that the data follows
# a normal distribution

# look at some histograms to verify normal distributions across
# the sensor data
data_processed_df[accel_cols + ["ex"]].plot.hist(
    by="ex", figsize=(20, 10), layout=(3, 3)
)

data_processed_df[gyro_cols + ["ex"]].plot.hist(
    by="ex", figsize=(20, 10), layout=(3, 3)
)

# mark with Chauvenet’s criterion
for col in outlier_cols:
    outlier_marked_df = mark_outliers_chauvenet(data_processed_df, col)
    plot_binary_outliers(
        outlier_marked_df, col, f"{col}_outlier", reset_index=True
    )

# can also try with a local outlier factor (distance based) function
# Unsupervised Outlier Detection using the Local Outlier Factor (LOF)
# note: LOF measures the local deviation of the density of a given
# sample with respect to its neighbors
# locality is given by k-nearest neighbors
#
# X_scores are certainty of outlier using the negative outlier factor
# i.e. more negative -> less chance of outlier
outliers_lof_ds, outliers, X_scores = mark_outliers_lof(
    data_processed_df, outlier_cols, n=20
)

for col in outlier_cols:
    plot_binary_outliers(outliers_lof_ds, col, "outlier_lof", reset_index=True)

# splitting the data (grouping by label) then checking outliers
label = "bench"
for col in outlier_cols:
    dataset = mark_outliers_chauvenet(
        data_processed_df[data_processed_df["ex"] == label],
        col,
    )
    plot_binary_outliers(dataset, col, f"{col}_outlier", reset_index=True)

# TODO pick an approach and filter out outliers for final dataset

# ------------------------------------------------------------ #
# for now we'll comment out because of periodic
# runs in jupyter interactive
# if __name__ == "__main__":
#     main()
