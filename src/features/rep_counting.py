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
class Exercise:
    def __init__(
        self, name: str, cutoff: float, pred_col: str = "accel_r"
    ) -> None:
        self.name = name
        self.cutoff = cutoff
        self.pred_col = pred_col

    def to_key_value_pair(self) -> tuple[str, "Exercise"]:
        return (self.name, self)


# ------------------------------------------------------------ #
def to_label_dict(df: pd.DataFrame, label_col: str) -> dict[str, pd.DataFrame]:
    return {
        label: df[df[label_col] == label] for label in df[label_col].unique()
    }


def cal_sum_of_squares_attr(
    df: pd.DataFrame, label_prefix: str
) -> pd.DataFrame:
    df_sum_sq: pd.DataFrame = df.copy()
    df_sum_sq[label_prefix + "_r"] = np.sqrt(
        df_minus_rest[label_prefix + "_x"] ** 2
        + df_minus_rest[label_prefix + "_y"] ** 2
        + df_minus_rest[label_prefix + "_z"] ** 2
    )

    return df_sum_sq


def calc_reps_naive(
    df: pd.DataFrame,
    col: str,
    lp_fs: float,
    lp_cutoff: float = 0.4,
    lp_order: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    filt = LowPassFilter.low_pass_filter(
        df,
        col=col,
        sampling_frequency=lp_fs,
        cutoff_frequency=lp_cutoff,
        order=lp_order,
    )

    indexes = argrelextrema(filt[col + "_lowpass"].values, np.less)
    rel_mins = df.iloc[indexes]
    return (filt, rel_mins)


def plot_rel_extrema_against_labels(
    df: pd.DataFrame, rel_extrema: pd.DataFrame, col: str
) -> None:
    lp_col = col + "_lowpass"
    fig, ax = plt.subplots()
    plt.plot(df[lp_col])
    plt.plot(rel_extrema[lp_col], "o", color="red")
    ax.set_ylabel(lp_col)
    effort = rel_extrema["effort"].iloc[0].title()
    exercise = rel_extrema["ex"].iloc[0].title()
    plt.title(f"{effort} {exercise}: {len(rel_extrema)} Reps")
    plt.show()


# ------------------------------------------------------------ #
# def main():
#     pass

df: pd.DataFrame = pd.read_pickle(Path(DATA_INTERIM_PATH, DATA_PKL_FILENAME))

# we dont care about rest periods when classifying reps
df_minus_rest: pd.DataFrame = df[df["ex"] != "rest"]

# injecting rep labels based on effort category
effort_map = {"heavy": 5, "medium": 10}
df_minus_rest["reps"] = df_minus_rest["effort"].apply(lambda x: effort_map[x])

# rep df to benchmark approach (ground-truth data)
rep_df: pd.DataFrame = (
    df_minus_rest.groupby(["ex", "effort", "set"])["reps"].max().reset_index()
)
rep_df["reps_pred"] = 0

# ------------------------------------------------------------ #
# sum of squares attributes
# note: we're attempting sum of squares because using r vs specific
# directions will allow us to be impartial to the device orientation,
# and can help with dynamic re-orientations

# magnitudes of accel & gyro columns
df_minus_rest = cal_sum_of_squares_attr(df_minus_rest, "accel")
df_minus_rest = cal_sum_of_squares_attr(df_minus_rest, "gyro")

# ------------------------------------------------------------ #
# splitting the data
label_dict = to_label_dict(df_minus_rest, "ex")

# ------------------------------------------------------------ #
# visualizing to identify patterns
plot_df: pd.DataFrame = label_dict["bench"]

# picking the first set for an example
plot_df[plot_df["set"] == plot_df["set"].unique()[0]].plot()

# ------------------------------------------------------------ #
# lowpass filtering

# using 75ms from pre-filtering -- can adjust this to 200ms
# later on to compare results -- 75ms seems to give some issues
# with correctly classifying bench -- errors as OHP
sampling_frequency = 1000 / 75

# testing lowpass filter
bench_df: pd.DataFrame = label_dict["bench"]
bench_set_ex = bench_df[bench_df["set"] == bench_df["set"].unique()[0]].drop(
    ["set"], axis=1
)

bench_set_ex["accel_r"].plot()

filt = LowPassFilter.low_pass_filter(
    bench_set_ex,
    col="accel_r",
    sampling_frequency=sampling_frequency,
    cutoff_frequency=0.4,
    order=10,
)

filt["accel_r_lowpass"].plot()

# ------------------------------------------------------------ #
# counting reps by calculating the relative extrema of data -- naive method
filt, extrema = calc_reps_naive(
    bench_set_ex, "accel_r", lp_fs=sampling_frequency
)

plot_rel_extrema_against_labels(filt, extrema, "accel_r")

# ------------------------------------------------------------ #
# predefined param values from testing
# these can be tuned with further testing
params = [
    Exercise("bench", 0.4),
    Exercise("squat", 0.35),
    Exercise("row", 0.65, pred_col="gyro_x"),
    Exercise("ohp", 0.35),
    Exercise("dead", 0.4),
]
ex_param_map = {x.name: x for x in params}

# ------------------------------------------------------------ #
# benchmarking the naive approach
for s in df_minus_rest["set"].unique():
    subset = df_minus_rest[df_minus_rest["set"] == s]
    ex_label = subset["ex"].iloc[0]
    exercise: Exercise = ex_param_map[ex_label]
    filt, extrema = calc_reps_naive(
        subset,
        exercise.pred_col,
        lp_fs=sampling_frequency,
        lp_cutoff=exercise.cutoff,
    )
    plot_rel_extrema_against_labels(filt, extrema, exercise.pred_col)


# ------------------------------------------------------------ #
# for now we'll comment out because of periodic
# runs in jupyter interactive
# if __name__ == "__main__":
#     main()
