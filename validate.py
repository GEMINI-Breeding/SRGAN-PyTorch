# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""File description: Realize the verification function after model training."""
import os

import torch
from torch.cuda import amp
from PIL import Image, ImageOps

from natsort import natsorted

import config
import imgproc
from model import Generator
from ssim import ssim
import numpy as np

def main() -> None:
    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize the super-resolution model
    print("Build SR model...")
    model = Generator().to(config.device)
    print("Build SR model successfully.")

    # Load the super-resolution model weights
    print(f"Load SR model weights `{os.path.abspath(config.model_path)}`...")
    state_dict = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(state_dict)
    print(f"Load SR model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the image evaluation index.
    total_psnr = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)
    
    lr_image_paths = natsorted(os.listdir(config.lr_dir))
    hr_image_paths = natsorted(os.listdir(config.hr_dir))

    for index in range(total_files):
        if 0:
            lr_image_path = os.path.join(config.lr_dir, file_names[index])
            sr_image_path = os.path.join(config.sr_dir, file_names[index])
            hr_image_path = os.path.join(config.hr_dir, file_names[index])
        else:
            lr_image_path = os.path.join(config.lr_dir, lr_image_paths[index])
            sr_image_path = os.path.join(results_dir, "SR_"+lr_image_paths[index])
            hr_image_path = os.path.join(config.hr_dir,hr_image_paths[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")

        # Resize lr
        if 1:
            width, height = hr_image.size
            newsize = (width//config.upscale_factor, height//config.upscale_factor)
            lr_image = lr_image.resize(newsize)

            width, height = hr_image.size
            newsize = (width-width%4, height-height%4)
            hr_image = hr_image.resize(newsize)

        if 1:
            # Conver to grayscale
            lr_image = ImageOps.grayscale(lr_image)
            hr_image = ImageOps.grayscale(hr_image)

        # Extract RGB channel image data
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

            
        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp_(0, 1)

        
        # Cal PSNR
        if 1:
            with amp.autocast():
                total_psnr += ssim(sr_tensor,hr_tensor)

        sr_image = imgproc.tensor2image(sr_tensor, range_norm=False, half=True)
        sr_image = np.reshape(sr_image,(sr_image.shape[0],sr_image.shape[1]))
        sr_image = Image.fromarray(sr_image)
        sr_image.save(sr_image_path)

    print(f"SSIM: {total_psnr / total_files:.2f}\n")


if __name__ == "__main__":
    main()
