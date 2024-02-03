from glob import glob
from pathlib import Path
import pandas as pd
from typing import Any


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

        # del columns that we don't care about
        to_del = ["epoch (ms)", "time (01:00)", "elapsed (s)"]
        for x in to_del:
            del agg_df[x]

    return agg_df


def read_set_from_df(df: pd.DataFrame, set_num: int) -> pd.DataFrame:
    return df[df["set"] == set_num]


def main() -> None:
    files = read_data()

    accel_files = [f for f in files if "_Accelerometer_" in f]
    # gyro measuring @ a higher frequency: we expect ~ 70% more records
    gyro_files = [f for f in files if "_Gyroscope_" in f]
    assert len(accel_files) > 0 and len(gyro_files) > 0

    # concat both dfs column-wise (axis=1)
    # we need to create a selection of the first df
    # -> use iloc[:,:3] (select first 3 columns) i.e the unique columns from accel files
    # the x/y/z acceleration
    accel_and_gyro_df = pd.concat(
        [
            aggregate_df_data(accel_files).iloc[:, :3],
            aggregate_df_data(gyro_files),
        ],
        axis=1,
    )

    # will result in a lot of NaN values because the accel & the gyro are
    # measuring at different frequencies in this dataset (todo: in the future
    # make sure both are measuring at the same frequency)
    #
    # as of now there will only be ~15% of rows that have full data because of random
    # time intervals where measurements happened at the same time between both sensors


if __name__ == "__main__":
    main()
