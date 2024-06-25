import os
import numpy as np
from PIL import Image
from functools import wraps
from sewar.full_ref import vifp
import torch
from torchvision import transforms
from tqdm import tqdm


def calculate_psnr(original_images, generated_images):
    mse = torch.mean((original_images - generated_images) ** 2, dim=[1, 2, 3])
    psnr = torch.where(
        mse == 0, float("inf"), 20 * torch.log10(255.0 / torch.sqrt(mse))
    )
    return psnr


def compute_vif_score(img1, img2):
    img1_np = np.array(transforms.ToPILImage()(img1))
    img2_np = np.array(transforms.ToPILImage()(img2))
    vif_score = vifp(img1_np, img2_np)
    if vif_score > 1.0:
        return 0.0
    return 1.0 - vif_score


def calculate_metrics_for_batch(original_images, transformed_images_list):
    batch_metrics = []
    for i, transformed_images in enumerate(transformed_images_list):
        psnr = calculate_psnr(original_images, transformed_images)
        vif = torch.zeros_like(psnr)
        for j in tqdm(
            range(original_images.shape[0]),
            desc=f"Calculating VIF for transform {i+1}/{len(transformed_images_list)}",
        ):
            vif[j] = compute_vif_score(
                original_images[j].cpu(), transformed_images[j].cpu()
            )
        batch_metrics.append((psnr.cpu().numpy(), vif.cpu().numpy()))
    return batch_metrics
