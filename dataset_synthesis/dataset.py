import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image


# Define the dataset class
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_path).convert("RGB")
        caption = os.path.splitext(self.file_list[idx])[
            0
        ]  # get the filename without extension

        if self.transform and isinstance(self.transform, transforms.Compose):
            image = self.transform(image)

        return image, caption


class ImageMetricsDataset(Dataset):
    def __init__(
        self, image_filenames, original_dir, transformed_dirs, suffixes, size=(512, 320)
    ):
        self.image_filenames = image_filenames
        self.original_dir = original_dir
        self.transformed_dirs = transformed_dirs
        self.suffixes = suffixes
        self.size = size
        self.resize = transforms.Resize(size)
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        base_filename = os.path.splitext(self.image_filenames[idx])[0]
        original_image = read_image(
            os.path.join(self.original_dir, self.image_filenames[idx])
        )
        original_image = self.to_tensor(self.resize(self.to_pil(original_image)))
        transformed_images = [
            self.to_tensor(
                self.resize(
                    self.to_pil(
                        read_image(
                            os.path.join(
                                transformed_dir, f"{base_filename}_{suffix}.jpg"
                            )
                        )
                    )
                )
            )
            for transformed_dir, suffix in zip(self.transformed_dirs, self.suffixes)
        ]
        return base_filename, original_image, transformed_images
