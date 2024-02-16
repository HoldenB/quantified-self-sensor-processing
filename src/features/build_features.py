import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
from DataTransformation import (
    LowPassFilter,
    PCA_Helper,
)
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation


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
DATA_PKL_FILENAME_FEATURES_OUT = "01_75ms_feature_extract_out.pkl"


# ------------------------------------------------------------ #
# def main():
#     pass

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
for s in set_vals:
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    duration = stop - start
    # select row for specific set and add a new column for duration
    # note: [(row predicate), new_col]
    df.loc[(df["set"] == s), "duration_s"] = duration.seconds

# average set duration
# group by exercise type and then project on only the duration
avg_rep_length_s = df.groupby(df["ex"])["duration_s"].mean()

# something to consider: How can we do this on the fly, in real time
# when we do not exactly know the full length of the set, if the buffer
# ends up cutting off before the set finishes or some duration afterwards,
# such that we end up with the set and a portion of rest in the buffer?

# something else to consider: could we learn the reps first, and then
# attempt to post-process and feature-extract the rough duration between
# reps to just "learn" the frequency?

# ------------------------------------------------------------ #
# low-pass filtering
df_lowpass: pd.DataFrame = df.copy()

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
df_pca: pd.DataFrame = df_lowpass.copy()
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

# need to further analyze the principal components to see if
# they could be potentially used for classification opposed to the
# original 6-dof values

# I'm not sure if PCA will really buy us anything here because
# even though angular acceleration does not have nearly as much
# variance as linear acceleration in most movements, the need to
# classify a curl vs hammer curl will rely solely on angular
# acceleration

# ------------------------------------------------------------ #
# sum of squares attributes
# note: we're attempting sum of squares because using r vs specific
# directions will allow us to be impartial to the device orientation,
# and can help with dynamic re-orientations
df_squared: pd.DataFrame = df_pca.copy()

# magnitude of the acceleration
accel_r = (
    df_squared["accel_x"] ** 2
    + df_squared["accel_y"] ** 2
    + df_squared["accel_z"] ** 2
)

# magnitude of gyro
gyro_r = (
    df_squared["gyro_x"] ** 2
    + df_squared["gyro_y"] ** 2
    + df_squared["gyro_z"] ** 2
)

df_squared["accel_r"] = np.sqrt(accel_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

# visualize another subset of the data
subset_5 = df_squared[df_squared["set"] == 14]
subset_5[["accel_r", "gyro_r"]].plot(subplots=True)

# ------------------------------------------------------------ #
# temporal abstraction - via rolling average / window
df_temporal: pd.DataFrame = df_squared.copy()

updated_predictor_cols = predictor_cols + ["accel_r", "gyro_r"]

# finding the window size (this will need to be tuned)
# window-size is our "look-back" size
# start off with a window size of 300ms
# note: our step size is 75ms
# step size = 500ms // 75ms = 6
window_size = 500 // 75

# need to compute mean & std for the updated predictor cols
# note: need to consider the case when the look-back window
# crosses over to the next exercise/set, so we need to
# project across the set space first before aggregation
df_temporal_sets = []
unique_sets = df_temporal["set"].unique()
for s in unique_sets:
    subset = df_temporal[df_temporal["set"] == s].copy()
    subset = NumericalAbstraction.abstract(
        # set-space projection of the data
        subset,
        updated_predictor_cols,
        window_size,
        aggregation_functions=["mean", "std"],
    )
    df_temporal_sets.append(subset)

# re-construct the temporal df from the subsets and replace the
# previous
df_temporal = pd.concat(df_temporal_sets)
# should notice 3 NaN values per set for each of the aggregation cols
df_temporal.info()

# example visualization
subset_6 = df_temporal_sets[4][
    ["accel_y", "accel_y_temp_mean_ws_6", "accel_y_temp_std_ws_6"]
].plot()

# ------------------------------------------------------------ #
# using DFT for frequency abstraction
# need to reset index because we expect a discrete
# representation of the data
df_freq: pd.DataFrame = df_temporal.copy().reset_index()

# num samples / sec i.e frequency of a sample
# (discrete sampling rate -- this needs to be an int)
sampling_rate = 1000 // 75
# window size = avg length of a rep
# we can tailor this using the list of averages we calculated earlier
# or for simplicity sake for now just make an approximation
# we can approximate ~2.8s or 2800ms per rep for now
# window_size = 2800 // 75

# note: using 14 here because of broadcasting error -- need to
# look further into this
window_size = 14

df_freq_ex: pd.DataFrame = FourierTransformation.abstract_frequency(
    df_freq, ["accel_y"], window_size, sampling_rate
)

df_freq_ex.columns

# visualizing a subset of this
subset_7 = df_freq_ex[df_freq_ex["set"] == 15]
subset_7[["accel_y"]].plot()
subset_7[
    [
        "accel_y_max_freq",
        "accel_y_freq_weighted",
        "accel_y_pse",
        "accel_y_freq_1.857_Hz_ws_14",
        "accel_y_freq_3.714_Hz_ws_14",
        "accel_y_freq_5.571_Hz_ws_14",
    ]
].plot()

# ------------------------------------------------------------ #
# splitting by the set and then running the DFT on each set
# for now we're doing this the exact same way as the temporal computation
df_freq_sets = []
unique_sets = df_freq["set"].unique()
for s in unique_sets:
    print(f"Applying Fourier transformation to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FourierTransformation.abstract_frequency(
        subset, updated_predictor_cols, window_size, sampling_rate
    )
    df_freq_sets.append(subset)

# re-construct the frequency df from the subsets and replace the
# previous
df_freq = pd.concat(df_freq_sets).set_index("epoch (ms)", drop=True)
df_freq.info()

# dealing with overlapping windows
# since we added a bunch of extra columns based on rolling windows
# this causes our data to be highly correlated on a row-by-row basis
# (adjacent rows)
#
# we need to avoid this to prevent overfitting later on when training
# need to allow for a certain % of overlap and then remove the rest of
# the data
df_freq = df_freq.dropna()

# typically we can allow for 50% -- meaning we can get rid of ~50% of
# the data
# this will ensure later models are less prone to overfitting
# we can tweak this later on and test with ranges of 50-80% overlap
# or variable based on the windowing size

# getting every other row
df_freq = df_freq.iloc[::2]

# ------------------------------------------------------------ #
# feature-engineering with k-means clustering
df_cluster: pd.DataFrame = df_freq.copy()

cluster_cols = ["accel_x", "accel_y", "accel_z"]
k_values = range(2, 10)
inertias = []

# testing the range of k-values
# we will plot the inertias & then use ELBOW
# technique to pick the optimal k-value
for k in k_values:
    subset = df_cluster[cluster_cols]
    k_means = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = k_means.fit_predict(X=subset)
    # inertia is the sum of squared distances of samples
    # to their closest cluster center, weighted by
    # the sample weights if provided
    #
    # we can treat inertia similar to principal components
    # when using the ELBOW method
    inertias.append(k_means.inertia_)


# plotting inertias
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("K-Value")
plt.ylabel("Sum of Sq. Dist. (Inertia)")
plt.show()

# result displays k=5 as the ELBOW however since we have 6 labels, we'll
# go with k=6 and try that out
elbow_k_result = 6
subset_km = df_cluster[cluster_cols]
k_means = KMeans(n_clusters=elbow_k_result, n_init=20, random_state=0)
cluster_labels = k_means.fit_predict(X=subset_km)
df_cluster["cluster"] = cluster_labels

# visualize the clusters to see if they make any sense
# need to eventually function all of this out...
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
unique_cluster_vals = df_cluster["cluster"].unique()
for cluster in unique_cluster_vals:
    subset = df_cluster[df_cluster["cluster"] == cluster]
    ax.scatter(
        subset["accel_x"], subset["accel_y"], subset["accel_z"], label=cluster
    )

ax.set_xlabel("x-accel")
ax.set_ylabel("y-accel")
ax.set_zlabel("z-accel")
plt.legend()
plt.show()

# now lets compare the same plot and instead of splitting by cluster
# we split by label (i.e. by exercise)
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
unique_label_vals = df_cluster["ex"].unique()
for label in unique_label_vals:
    subset = df_cluster[df_cluster["ex"] == label]
    ax.scatter(
        subset["accel_x"], subset["accel_y"], subset["accel_z"], label=label
    )

ax.set_xlabel("x-accel")
ax.set_ylabel("y-accel")
ax.set_zlabel("z-accel")
plt.legend()
plt.show()

# exporting the pkl
df_cluster.to_pickle(Path(DATA_INTERIM_PATH, DATA_PKL_FILENAME_FEATURES_OUT))

# ------------------------------------------------------------ #
# for now we'll comment out because of periodic
# runs in jupyter interactive
# if __name__ == "__main__":
#     main()
