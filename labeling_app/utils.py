import os
import pandas as pd
from PIL import Image
import csv
from pathlib import Path

CSV_FILENAME = "labels.csv"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def create_csv_if_not_exist():
    if not Path(CSV_FILENAME).is_file():
        with open(CSV_FILENAME, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "Label"])


def get_image_files(image_dir):
    return [
        f.name
        for f in Path(image_dir).iterdir()
        if f.is_file() and f.name.lower().endswith(IMAGE_EXTENSIONS)
    ]


def load_labels():
    return pd.read_csv(CSV_FILENAME)


def update_labels(df_labels, image_file, selected_label):
    if image_file in df_labels["Image"].values:
        df_labels.loc[df_labels["Image"] == image_file, "Label"] = selected_label
    else:
        df_labels = df_labels.append(
            {"Image": image_file, "Label": selected_label},
            ignore_index=True,
        )
    df_labels.to_csv(CSV_FILENAME, index=False)
