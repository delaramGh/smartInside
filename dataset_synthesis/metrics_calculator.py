import os
import random
import pandas as pd
from tqdm import tqdm
from dataset import ImageMetricsDataset
from torch.utils.data import DataLoader
from metrics_utilities import calculate_metrics_for_batch


if __name__ == "__main__":
    # Path to the directories with your original and transformed images
    original_dir = (
        "/Users/alimirferdos/Research/2021AIChamp_Submission/data/11.한국전력공사_데이터/train/"
    )
    root_output_folder = (
        "/Users/alimirferdos/Research/2021AIChamp_Submission/data/generated_data/rain"
    )
    albumentations_transformed = os.path.join(root_output_folder, "albumentation")
    torchvision_transformed = os.path.join(root_output_folder, "torchvision")

    albumentations_suffix = "albumentations"
    torchvision_suffix = "torchvision"

    # Get a list of all original images
    original_images_filenames = os.listdir(original_dir)

    # Define the sample size
    # sample_size = 10  # Change this to the number of images you want to sample

    # Sample a subset of the original images
    # original_images_sample = random.sample(original_images, sample_size)
    original_images_sample = original_images_filenames

    # Create the Dataset
    dataset = ImageMetricsDataset(
        image_filenames=original_images_sample,
        original_dir=original_dir,
        transformed_dirs=[albumentations_transformed, torchvision_transformed],
        suffixes=[albumentations_suffix, torchvision_suffix],
    )

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=32)

    # Prepare an empty list to collect all the metrics
    metrics = []

    # Loop over the batches
    for base_filenames, original_images, transformed_images_list in tqdm(
        dataloader, colour="green"
    ):
        # Calculate the metrics for this batch and add them to the list
        batch_metrics = calculate_metrics_for_batch(
            original_images, transformed_images_list
        )
        for i, (psnr, vif) in enumerate(batch_metrics):
            metrics.extend(
                [
                    (
                        filename,
                        "albumentations" if i == 0 else "torchvision",
                        psnr_item,
                        vif_item,
                    )
                    for filename, psnr_item, vif_item in zip(base_filenames, psnr, vif)
                ]
            )

    # Convert the list of metrics to a DataFrame
    df_metrics = pd.DataFrame(metrics, columns=["image", "dataset", "psnr", "vif"])

    # Save the DataFrame to a CSV file
    df_metrics.to_csv("metrics.csv", index=False)
