import os
import random
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm


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
original_images = os.listdir(original_dir)

# Define the sample size
sample_size = 50  # Change this to the number of images you want to sample

# Sample a subset of the original images
original_images_sample = random.sample(original_images, sample_size)

# Prepare an empty list to collect all the frames for the gif
frames = []

for filename in tqdm(original_images_sample, colour="green"):
    # Remove the extension from the filename
    base_filename = os.path.splitext(filename)[0]

    # Open the original image
    original_image = np.array(Image.open(os.path.join(original_dir, filename)))

    # Find and open the corresponding transformed images
    transformed_image_1 = np.array(
        Image.open(
            os.path.join(
                albumentations_transformed,
                f"{base_filename}_{albumentations_suffix}.jpg",
            )
        )
    )
    transformed_image_2 = np.array(
        Image.open(
            os.path.join(
                torchvision_transformed, f"{base_filename}_{torchvision_suffix}.jpg"
            )
        )
    )

    # Create a new figure and set its size
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display the original and transformed images side by side
    axs[0].imshow(original_image)
    axs[0].set_title("Original")

    axs[1].imshow(transformed_image_1)
    axs[1].set_title("Albumentations Transformed")

    axs[2].imshow(transformed_image_2)
    axs[2].set_title("InstructPix2Pix Diffusion Transformed")

    # Save the figure to a temporary .png file
    fig.savefig("temp.png")

    # Open the saved image and add it to the frames list
    frames.append(imageio.imread("temp.png"))

# Create the gif using the frames list
imageio.mimsave("transformed_images.gif", frames, "GIF", duration=1)
