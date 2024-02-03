from glob import glob
from pathlib import Path
import pandas as pd


def read_data():
    data_path = Path("../data")
    files = glob(f"{data_path}/*.csv")
    assert len(files) > 0

    return files


def parse_labels_from_filename(filename: str):
    ex_category_labels = ["participant", "ex", "effort"]
    device_category_labels = [
        "device",
        "utc_dateTime",
        "device_id",
        "sensor_type",
        "frequency",
    ]

    file_cleaned: list[str] = filename.split("/")[2:][0].split("_")[-:1]
    assert len(file_cleaned) > 1 and len(file_cleaned) - 1 == len(
        device_category_labels
    )

    ex_categories = [*zip(ex_category_labels, file_cleaned[0].split("-"))]
    device_categories = [*zip(device_category_labels, file_cleaned[1:])]
    # also need to clean up effort labels:
    # heavy3 (1, 2, 3) -> heavy (additional processing after zip)
    effort_label, effort_val = ex_categories.pop()
    ex_categories.append((effort_label, effort_val.rstrip("123")))

    return ex_categories + device_categories


def aggregate_df_data(files: list[str]):
    agg_df = pd.DataFrame()
    # oof probably a ton of mem usage but oh well
    # we can be better and stream this in the future
    for i, file in enumerate(files):
        labels = parse_labels_from_filename(file)
        file_df = pd.read_csv(file)
        for label, value in labels:
            file_df[label] = value

        agg_df = pd.concat([agg_df, file_df])

    return agg_df


def main() -> None:
    files = read_data()

    accel_files = [f for f in files if "_Accelerometer_" in f]
    # gyro measuring @ a higher frequency: we expect ~ 70% more records
    gyro_files = [f for f in files if "_Gyroscope_" in f]
    assert len(accel_files) > 0 and len(gyro_files) > 0

    accel_df = aggregate_df_data(accel_files)
    gyro_df = aggregate_df_data(gyro_files)


if __name__ == "__main__":
    main()
