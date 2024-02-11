import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from DataTransformation import (
    LowPassFilter,
    PCA_Helper,
)
from TemporalAbstraction import NumericalAbstraction


# ------------------------------------------------------------ #
# rcParams
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# ------------------------------------------------------------ #
LEGEND_STYLE_KWARGS = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, 1.15),
    "ncols": 3,
    "fancybox": True,
    "shadow": True,
}

DATA_PATH = Path("../../data")
DATA_INTERIM_PATH = Path(DATA_PATH, "interim")
DATA_PKL_FILENAME_OUTLIERS_REM = "01_75ms_outliers_removed_chauvenet.pkl"


# ------------------------------------------------------------ #
df: pd.DataFrame = pd.read_pickle(
    Path(DATA_INTERIM_PATH, DATA_PKL_FILENAME_OUTLIERS_REM)
)
df.info()
predictor_cols = list(df.columns[:6])

# example of a discontinuous set due to outliers
subset_1 = df[df["set"] == 35]["gyro_y"]
subset_1.plot()

# ------------------------------------------------------------ #
# need to use imputation to deal with the missing values (NaN)
# values that have been previously removed with outlier detection
# with imputation we will use interpolation to fill in the gaps

# linear interpolation (surprisingly works really well)
interpolated_subset = subset_1.interpolate()
interpolated_subset.plot()

# interpolate all the things
for col in predictor_cols:
    df[col] = df[col].interpolate()
df.info()

# ------------------------------------------------------------ #
subset_2 = df[df["set"] == 27]["accel_y"]
subset_2.plot()

# we need to know how long a specific rep takes, this way we can
# tune frequency settings and allow smoothening/filtering to adjust
# to specific reps -> higher frequency for faster reps
# the goal is to attempt at feature engineering a set duration

# example of calculating the set duration based on the difference in timestamps
(df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]).seconds

# set duration for all sets
set_vals = df["set"].unique()
for set in set_vals:
    start = df[df["set"] == set].index[0]
    stop = df[df["set"] == set].index[-1]
    duration = stop - start
    # select row for specific set and add a new column for duration
    # note: [(row predicate), new_col]
    df.loc[(df["set"] == set), "duration_s"] = duration.seconds

# average set duration
# group by exercise type and then project on only the duration
df.groupby(df["ex"])["duration_s"].mean()

# something to consider: How can we do this on the fly, in real time
# when we do not exactly know the full length of the set, if the buffer
# ends up cutting off before the set finishes or some duration afterwards,
# such that we end up with the set and a portion of rest in the buffer?

# something else to consider: could we learn the reps first, and then
# attempt to post-process and feature-extract the rough duration between
# reps to just "learn" the frequency?

# ------------------------------------------------------------ #
# low-pass filtering
df_lowpass = df.copy()

# sampling frequency = step size between reps
# cutoff frequency = tuning param - look @ filter results and consider
# avg rep duration

# recall the step size that we re-processed the sensor values at was
# 75ms, so sampling freq = 1000ms / 75ms = ~13.3 instances/sec
sample_freq = 1000 / 75

# critical to tune this
# we can potentially auto-tune
cutoff_freq = 1.8

lp_result = LowPassFilter.low_pass_filter(
    df_lowpass, "accel_y", sample_freq, cutoff_freq
)

# comparing preprocessed to lowpass
# select subset to compare
subset_3 = df_lowpass[df_lowpass["set"] == 43]

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset_3["accel_y"].reset_index(drop=True), label="Raw Data")
ax[1].plot(
    subset_3["accel_y_lowpass"].reset_index(drop=True),
    label="Butterworth Lowpass Filter",
)
ax[0].legend(**LEGEND_STYLE_KWARGS)
ax[1].legend(**LEGEND_STYLE_KWARGS)

# applying lowpass filter to all predictor columns
for col in predictor_cols:
    df_lowpass = LowPassFilter.low_pass_filter(
        df_lowpass, col, sample_freq, cutoff_freq
    )
    # we can add lpf as additional cols but for now we'll
    # override the original cols
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    df_lowpass = df_lowpass.drop(col + "_lowpass", axis=1)

# ------------------------------------------------------------ #
# PCA
df_pca = df_lowpass.copy()
pca_helper = PCA_Helper()

# analyze before determining the number of components we'll use
# note: using Elbow technique to determine optimal component number
# Elbow evaluates the components that capture the most variance up
# until the rate of change in variance diminishes (i.e at the Elbow)
pca_variance = pca_helper.determine_pc_explained_variance(
    df_pca, predictor_cols
)

# visualizing the explained variance as correlated to the number
# of principal components
# i.e. as we increase the number of principal components, how much
# do the latter components still contribute to the variance
# so how much information do the components carry that is viable for us

# visually see the elbow around 3
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_cols) + 1), pca_variance)
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance")
plt.show()

df_pca = pca_helper.apply_pca(df_pca, predictor_cols, number_comp=3)
# visualize the principal components
subset_4 = df_pca[df_pca["set"] == 35]
subset_4[["pca_1", "pca_2", "pca_3"]].plot()

# ------------------------------------------------------------ #
