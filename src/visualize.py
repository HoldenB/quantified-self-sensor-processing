import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from IPython.display import display
from typing import NamedTuple


# adjusting plot settings with runtime rcParams
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (15, 5)
mpl.rcParams["figure.dpi"] = 100


# ------------------------------------------------------------ #
LEGEND_STYLE_KWARGS = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, 1.15),
    "ncols": 3,
    "fancybox": True,
    "shadow": True,
}

DATA_PATH = Path("../data")
DATA_INTERIM_PATH = Path(DATA_PATH, "interim")
DATA_PKL_FILENAME = "01_75ms_data_processed.pkl"

REPORTS_PATH = Path("../reports")
FIGURES_PATH = Path(REPORTS_PATH, "figures")


# ------------------------------------------------------------ #
class Range(NamedTuple):
    low: int
    high: int


def create_figure_filepath(
    ex: str,
    participant: str,
    version: str = "v1.0",
    file_ext: str = "png",
) -> Path:
    return Path(
        FIGURES_PATH, f"{version}__{ex}__{participant}__accel_gyro.{file_ext}"
    )


def plot_subsets_by_ex_label(
    data_df: pd.DataFrame,
    range: Range = None,
    float_label: str = "accel_y",
) -> None:
    # unique exercise values w/in cols
    ex_labels = data_df["ex"].unique()
    for label in ex_labels:
        subset: pd.DataFrame = data_df[data_df["ex"] == label]
        # display(subset.head(4))
        fig, ax = plt.subplots()
        subset_with_range = (
            subset[:] if not range else subset[range.low : range.high]
        )
        plt.plot(
            subset_with_range[float_label].reset_index(drop=True),
            label=label,
        )
        plt.legend()
        plt.show()


def grouped_plot_effort_for_ex(
    data_df: pd.DataFrame,
    ex: str = "squat",
    participant: str = "A",
    float_label: str = "accel_y",
    y_label="y-acceleration (m/s)",
) -> None:
    # comparing based on set effort i.e medium vs heavy sets
    # can we spot a difference between these sets for a given exercise?
    # quicker way to query subsets of a df -> note: string match needs single quotes within
    # note: reset index to avoid lapses in time -> index by sample num (N)
    category_df = (
        data_df.query(expr=f"ex == '{ex}'")
        .query(expr=f"participant == '{participant}'")
        .reset_index()
    )

    # need to create a grouped plot that is limited to single col (accel_y)
    # we can call plot directly on a pd dataframe
    # this will allow us to visualize difference in effort by color
    fix, ax = plt.subplots()
    category_df.groupby(["effort"])[float_label].plot()
    ax.set_xlabel("sample (n)")
    ax.set_ylabel(y_label)
    plt.legend()


def multi_ax_grouped_plot_for_ex(
    data_df: pd.DataFrame,
    float_label_y_axes=["accel_x", "accel_y", "accel_z"],
) -> None:
    ex_labels = data_df["ex"].unique()
    participants = data_df["participant"].unique()
    combinations = [(ex, part) for ex in ex_labels for part in participants]

    for comb in combinations:
        mult_axes_df = (
            data_df.query(expr=f"ex == '{comb[0]}'")
            .query(expr=f"participant == '{comb[1]}'")
            .reset_index()
        )

        # filter out empty data (i.e. part D did not do ohp)
        if len(mult_axes_df) == 0:
            continue

        fix, ax = plt.subplots()
        mult_axes_df[float_label_y_axes].plot(ax=ax)
        ax.set_xlabel("samples (n)")
        plt.title(f"{comb[0]} | {comb[1]}".title())
        plt.legend()
        plt.show()


# TODO I know, I know...a lot needs to be functioned out and shared
def multi_ax_grouped_plot_for_accel_gyro_for_ex(
    data_df: pd.DataFrame,
    save_figs=False,
) -> None:
    ex_labels = data_df["ex"].unique()
    participants = data_df["participant"].unique()
    combinations = [(ex, part) for ex in ex_labels for part in participants]

    for comb in combinations:
        mult_axes_df = (
            data_df.query(expr=f"ex == '{comb[0]}'")
            .query(expr=f"participant == '{comb[1]}'")
            .reset_index(drop=True)
        )

        # filter out empty data (i.e. part D did not do ohp)
        if len(mult_axes_df) == 0:
            continue

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
        fig.suptitle(f"{comb[0]} | {comb[1]}".title(), fontsize=12)

        mult_axes_df[["accel_x", "accel_y", "accel_z"]].plot(ax=ax[0])
        mult_axes_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])

        ax[0].set_ylabel("acceleration (g)")
        ax[1].set_ylabel("angular velocity (deg/s)")

        # styling
        ax[0].legend(**LEGEND_STYLE_KWARGS)
        ax[1].legend(**LEGEND_STYLE_KWARGS)
        ax[1].set_xlabel("samples (n)")

        if save_figs:
            plt.savefig(create_figure_filepath(comb[0], comb[1]))

        plt.show()


# ------------------------------------------------------------ #
# def main():
#     pass

data_processed_df: pd.DataFrame = pd.read_pickle(
    Path(DATA_INTERIM_PATH, DATA_PKL_FILENAME)
)

# looking at individual sets - subset of our processed df
set_df: pd.DataFrame = data_processed_df[data_processed_df["set"] == 1]

# plotting a single column
# temp get rid of timestamp -> conv index to integer
# plt.plot(set_df["accel_y"].reset_index(drop=True))

# ------------------------------------------------------------ #
plot_subsets_by_ex_label(
    data_processed_df, range=Range(0, 100), float_label="accel_y"
)

# ------------------------------------------------------------ #
grouped_plot_effort_for_ex(
    data_processed_df,
    ex="squat",
    participant="A",
    float_label="accel_y",
    y_label="y-acceleration (g)",
)

# ------------------------------------------------------------ #
# compare participants -> reset index to remove index by epoch and just index
# by samples (sample #) b/c the times that participants performed the sets
# could be intertwined
participant_df = (
    data_processed_df.query(expr="ex == 'bench'")
    .sort_values(by="participant")
    .reset_index()
)

fix, ax = plt.subplots()
participant_df.groupby(["participant"])["accel_y"].plot()
ax.set_xlabel("samples (n)")
ax.set_ylabel("y-acceleration (g)")
plt.legend()

# ------------------------------------------------------------ #
# plotting multiple axes
# visualizing x/y/z linear and angular acceleration
ex_mult = "squat"
participant_mult = "A"
mult_axes_df = (
    data_processed_df.query(expr=f"ex == '{ex_mult}'")
    .query(expr=f"participant == '{participant_mult}'")
    .reset_index()
)

# note:
# type(df["label"])     -> series (single-col)
# type(df[["label"]])   -> dataframe (multi-col)

fix, ax = plt.subplots()
mult_axes_df[["accel_x", "accel_y", "accel_z"]].plot(ax=ax)
ax.set_xlabel("samples (n)")
ax.set_ylabel("y-acceleration (g)")
plt.legend()

# ------------------------------------------------------------ #
# plotting all combinations per sensor
# multi_ax_grouped_plot_for_ex(
#     data_processed_df, float_label_y_axes=["accel_x", "accel_y", "accel_z"]
# )
# multi_ax_grouped_plot_for_ex(
#     data_processed_df, float_label_y_axes=["gyro_x", "gyro_y", "gyro_z"]
# )

# ------------------------------------------------------------ #
# combining accel & gyro data for a specific participant and exercise
multi_ax_grouped_plot_for_accel_gyro_for_ex(data_processed_df, save_figs=False)

# ------------------------------------------------------------ #
# for now we'll comment out because of periodic
# runs in jupyter interactive
# if __name__ == "__main__":
#     main()
