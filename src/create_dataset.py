from glob import glob
from pathlib import Path
import pandas as pd
from typing import Any

DATA_PATH = Path("../data")
DATA_INTERIM_PATH = Path(DATA_PATH, "interim")


def read_data():
    data_path = Path("../data")
    files = glob(f"{data_path}/*.csv")
    assert len(files) > 0

    return files


def parse_labels_from_filename(
    filename: str,
) -> list[tuple[tuple[str, Any], Any]]:
    ex_category_labels = ["participant", "ex", "effort"]
    device_category_labels = [
        "device",
        "utc_dateTime",
        "device_id",
        "sensor_type",
        "frequency",
    ]

    # need to include some transformation functions for epoch and dateTime
    # should match the index in the label list (so we can zip them all)
    # note: pd.df transformations
    category_df_transformations = [
        lambda df, label: pd.Series(df[label], dtype="string"),  # participant
        lambda df, label: pd.Series(df[label], dtype="string"),  # ex
        lambda df, label: pd.Series(df[label], dtype="string"),  # effort
        lambda df, label: pd.Series(df[label], dtype="string"),  # device
        lambda df, label: pd.to_datetime(df[label]),  # utc_dateTime
        lambda df, label: pd.Series(df[label], dtype="string"),  # device_id
        lambda df, label: pd.Series(df[label], dtype="string"),  # sensor_type
        lambda df, label: pd.Series(df[label], dtype="string"),  # frequency
    ]

    file_cleaned: list[str] = filename.split("/")[2:][0].split("_")[:-1]
    assert len(file_cleaned) > 1 and len(file_cleaned) - 1 == len(
        device_category_labels
    )

    ex_categories = [*zip(ex_category_labels, file_cleaned[0].split("-"))]
    device_categories = [*zip(device_category_labels, file_cleaned[1:])]

    # also need to clean up effort labels:
    # heavy3 (1, 2, 3) -> heavy (additional processing after zip)
    assert len(ex_categories) > 0
    effort_label, effort_val = ex_categories.pop()
    ex_categories.append((effort_label, effort_val.rstrip("123")))

    categories_and_transforms = [
        *zip(ex_categories + device_categories, category_df_transformations)
    ]

    # add any additional exclusionary labels here if needed
    # note: dateTime is already captured in the files: not needed from filename
    exclusionary_labels = ["utc_dateTime"]
    return [
        *filter(
            # ie ((label, value), transformation) -> x[0][0]
            lambda x: x[0][0] not in exclusionary_labels,
            categories_and_transforms,
        )
    ]


def aggregate_df_data(files: list[str]) -> pd.DataFrame:
    agg_df = pd.DataFrame()
    # oof probably a ton of mem usage but oh well
    # we can be better and stream this in the future
    for i, file in enumerate(files):
        labels = parse_labels_from_filename(file)
        file_df = pd.read_csv(file)
        file_df["set"] = i + 1

        for (label, value), transform in labels:
            file_df[label] = value
            assert type(file_df) == pd.DataFrame
            # apply category df transformation for each label
            file_df[label] = transform(file_df, label)

        agg_df = pd.concat([agg_df, file_df])

        # post-processing
        # need to convert df into time-series df so we can
        # re-sample later on
        # override the index with epoch column
        epoch_key = "epoch (ms)"
        agg_df.index = pd.to_datetime(agg_df[epoch_key], unit="ms")

    return agg_df


def read_set_from_df(df: pd.DataFrame, set_num: int) -> pd.DataFrame:
    return df[df["set"] == set_num]


def main() -> None:
    files = read_data()

    accel_files = [f for f in files if "_Accelerometer_" in f]
    # gyro measuring @ a higher frequency: we expect ~ 70% more records
    gyro_files = [f for f in files if "_Gyroscope_" in f]
    assert len(accel_files) > 0 and len(gyro_files) > 0

    accel_df = aggregate_df_data(accel_files)
    gyro_files = aggregate_df_data(gyro_files)

    # concat both dfs column-wise (axis=1)
    # we need to create a selection of the first df
    # -> use iloc[:,:3] (select first 3 columns) i.e the unique columns from accel files
    # the x/y/z acceleration
    accel_and_gyro_df = pd.concat(
        [
            # select cols 3-10 (inclusive)
            accel_df.iloc[:, 3:10],
            gyro_files.iloc[:, 3:6],
        ],
        axis=1,
    )

    accel_and_gyro_df = accel_and_gyro_df.sort_values(by="epoch (ms)")

    # will result in a lot of NaN values because the accel & the gyro are
    # measuring at different frequencies in this dataset (todo: in the future
    # make sure both are measuring at the same frequency)
    #
    # as of now there will only be ~15% of rows that have full data because of random
    # time intervals where measurements happened at the same time between both sensors
    # can visualize this with df.head(50)

    updated_col_names = [
        "accel_x",
        "accel_y",
        "accel_z",
        "set",
        "participant",
        "ex",
        "effort",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ]

    # rename cols
    accel_and_gyro_df.columns = updated_col_names

    # need to reduce frequency between each measurement (so we have data for
    # each row)
    # sampling:
    # using s = 1/T where T is the period
    # Accelerometer:    12.500HZ -> measure every 1/25      -> 0.04s
    # Gyroscope:        25.000HZ -> measure every 1/12.5    -> 0.08s

    # resampling the time-series data according to the frequency difference:
    # (sync up frequency between both sensors)
    # see: https://pandas.pydata.org/docs/user_guide/timeseries.html#sparse-resampling

    # need to apply aggregation rules to preserve categorical columns first
    sampling_rules_for_cols = [
        "mean",
        "mean",
        "mean",
        "last",  # categorical
        "last",  # categorical
        "last",  # categorical
        "last",  # categorical
        "mean",
        "mean",
        "mean",
    ]

    rules = dict([*zip(updated_col_names, sampling_rules_for_cols)])

    # in order to save on compute, we'll split by frequency=day
    # and process each day, dropping NA values one day at a time
    data_split_by_day = [
        group for n, group in accel_and_gyro_df.groupby(pd.Grouper(freq="D"))
    ]

    data_resampled: pd.DataFrame = pd.concat(
        [df.resample("75ms").apply(rules).dropna() for df in data_split_by_day]
    )

    # fix set column -> float64 -> int
    data_resampled["set"] = data_resampled["set"].astype("int")

    # export the data as .pkl
    # using pkl for interim files so we can easily just read it in from the processing module
    data_resampled.to_pickle(Path(DATA_INTERIM_PATH, "01_data_processed.pkl"))


if __name__ == "__main__":
    main()
