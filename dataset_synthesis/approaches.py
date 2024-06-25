import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
import torch
from typing import List
from diffusers import StableDiffusionInstructPix2PixPipeline
from threading import Timer


# Define the base Approach class
class Approach:
    def __init__(self, transform=None):
        self.transform = self._set_transform(transform)
        self.save_file_suffix = "transformed"

    def _set_transform(self, transform):
        if transform is None:
            return transforms.Compose(
                [
                    transforms.Resize((320, 512)),
                    transforms.ToTensor(),
                ]
            )
        return transform

    def visualize_transformed_images(
        self, original_image, albumentations_image, torchvision_image, timer=0.1
    ):
        self._display_images(original_image, albumentations_image, torchvision_image)
        self._auto_close_plot(timer=timer)

    def handle_batch(self, batch, output_folder):
        pass

    def save_transformed_image(self, image, output_folder, original_filename):
        self._create_directory_if_not_exists(output_folder)

        output_path = self._get_output_path(output_folder, original_filename)
        image.save(output_path)

    def _display_images(self, original_image, albumentations_image, torchvision_image):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        ax[0].imshow(np.array(original_image))
        ax[0].set_title("Original Image")

        if albumentations_image is not None:
            ax[1].imshow(np.array(albumentations_image))
            ax[1].set_title("Albumentations Transformed")

        if torchvision_image is not None:
            ax[2].imshow(np.array(torchvision_image))
            ax[2].set_title("Torchvision Transformed")

        plt.tight_layout()
        plt.pause(0.1)
        plt.show(block=False)

    def _auto_close_plot(self, timer=0.1):
        # Set a timer to close the figure.
        t = Timer(timer, plt.close)
        t.start()

    def _create_directory_if_not_exists(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def _get_output_path(self, output_folder, original_filename):
        # Use consistent naming scheme for output images
        return os.path.join(
            output_folder, f"{original_filename}_{self.save_file_suffix}.jpg"
        )


# Define the Torchvision Approach class
class TorchvisionApproach(Approach):
    def __init__(self, transformation_prompt, negative_prompt, transform=None):
        super().__init__(transform=transform)
        self.transformation_prompt = transformation_prompt
        self.negative_prompt = negative_prompt
        self.save_file_suffix = "torchvision"
        self.pipe = self._load_model()

    def handle_batch(self, batch, output_folder):
        input_images, captions = batch
        transformed_images = self.batch_generate(
            input_images, self.transformation_prompt, self.negative_prompt, steps=10
        )

        for i in range(len(input_images)):
            self.visualize_transformed_images(
                transforms.ToPILImage()(input_images[i]),
                None,
                transformed_images[i],
                timer=1,
            )
            self.save_transformed_image(
                transformed_images[i], output_folder, captions[i]
            )

    def batch_generate(
        self,
        input_images: List[Image.Image],
        instruction: str,
        negative_instruction: str,
        steps: int = 200,
        randomize_cfg: bool = True,
    ) -> List[Image.Image]:
        seed = random.randint(0, 2**32 - 1)
        text_cfg_scale, image_cfg_scale = self._get_random_config_scales(randomize_cfg)
        transformed_images = []

        for input_image in input_images:
            input_image = transforms.ToPILImage()(input_image)

            if instruction == "":
                transformed_images.append(input_image)
                continue

            generator = torch.manual_seed(seed)
            edited_image = self.pipe(
                instruction,
                negative_prompt=negative_instruction,
                image=input_image,
                guidance_scale=text_cfg_scale,
                image_guidance_scale=image_cfg_scale,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]

            transformed_images.append(edited_image)

        return transformed_images

    def _load_model(self):
        # Load the model into the class
        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, safety_checker=None
        )
        return pipe.to("mps")

    def _get_random_config_scales(self, randomize_cfg):
        # Generate random configuration scales if requested
        text_cfg_scale = (
            round(random.uniform(6.0, 9.0), ndigits=2)
            if randomize_cfg
            else text_cfg_scale
        )
        image_cfg_scale = (
            round(random.uniform(1.2, 1.8), ndigits=2)
            if randomize_cfg
            else image_cfg_scale
        )
        return text_cfg_scale, image_cfg_scale


# Define the Albumentations Approach class
class AlbumentationsApproach(Approach):
    def __init__(self, transformation_pipeline, transform=None):
        super().__init__(transform=transform)
        self.transformation_pipeline = transformation_pipeline
        self.save_file_suffix = "albumentations"

    def handle_batch(self, batch, output_folder):
        input_images, captions = batch
        transformed_images = self._batch_generate(input_images, captions, output_folder)

        for i in range(len(input_images)):
            self.visualize_transformed_images(
                transforms.ToPILImage()(input_images[i]), transformed_images[i], None
            )
            self.save_transformed_image(
                transforms.ToPILImage()(transformed_images[i]),
                output_folder,
                captions[i],
            )

    def _batch_generate(self, input_images, captions, output_folder):
        transformed_images = []

        for input_image in input_images:
            original_image = np.array(transforms.ToPILImage()(input_image))
            transformed_image = self.transformation_pipeline(image=original_image)[
                "image"
            ]
            transformed_images.append(transformed_image)

        return transformed_images
