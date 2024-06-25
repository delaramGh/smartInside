import os
import argparse
from torch.utils.data import DataLoader
import albumentations as A
from dataset import ImageDataset
from approaches import TorchvisionApproach, AlbumentationsApproach
from tqdm import tqdm


# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--approach",
    type=int,
    default=2,
    help="1 for torchvision approach, 2 for albumentations approach",
)
args = parser.parse_args()


def main():
    dataset_path = (
        "/Users/alimirferdos/Research/2021AIChamp_Submission/data/11.한국전력공사_데이터/train/"
    )
    output_folder = (
        "/Users/alimirferdos/Research/2021AIChamp_Submission/data/generated_data/rain"
    )
    output_folder = os.path.join(
        output_folder, "albumentation" if args.approach == 2 else "torchvision"
    )

    if args.approach == 1:
        approach = TorchvisionApproach("add rain")
        batch_size = 1
    elif args.approach == 2:
        tranformation_pipeline = A.Compose(
            [
                A.augmentations.geometric.resize.Resize(640, 1024),
                A.RandomRain(blur_value=7, brightness_coefficient=1, always_apply=True),
            ]
        )
        approach = AlbumentationsApproach(
            transformation_pipeline=tranformation_pipeline
        )
        batch_size = 1

    # Create an instance of the ImageDataset
    image_dataset = ImageDataset(dataset_path, transform=approach.transform)

    # Create a DataLoader to handle batching and shuffling
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)

    # Iterate over the data loader to access images in batches
    for batch in tqdm(data_loader, colour="green"):
        approach.handle_batch(batch, output_folder)


if __name__ == "__main__":
    main()
