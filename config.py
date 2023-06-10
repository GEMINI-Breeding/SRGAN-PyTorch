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
"""Realize the parameter configuration function of dataset, model, training and verification code."""
from tkinter.font import families
from numpy import Infinity
import torch
from torch.backends import cudnn as cudnn
import sys

class Config():

    def __init__(self, mode="train_srgan",exp_name="test"):
        # ==============================================================================
        # General configuration
        # ==============================================================================
        # Random seed to maintain reproducible results
        torch.manual_seed(0)
        # Use GPU for training by default
        self.device = torch.device("cuda", 0)
        # Turning on when the image size does not change during training can speed up training
        cudnn.benchmark = True
        # Image magnification factor
        self.upscale_factor = 4
        # Current configuration parameter method
        self.mode = mode
        #mode = "train_srgan"

        # Experiment name, easy to save weights and log files
        self.exp_name = exp_name

        # ==============================================================================
        # Training SRResNet model configuration
        # ==============================================================================
        if self.mode == "train_srresnet":
            # Dataset address
            self.train_image_dir = "data/ImageNet/SRGAN/train"
            self.valid_image_dir = "data/ImageNet/SRGAN/valid"

            self.image_size = 96
            self.batch_size = 16
            self.num_workers = 4

            # Incremental training and migration training
            self.resume = False
            self.strict = False
            self.start_epoch = 0
            self.resume_weight = ""

            # Total num epochs
            self.epochs = 0

            # Adam optimizer parameter for SRResNet(p)
            self.model_lr = 1e-4
            self.model_betas = (0.9, 0.999)

            # Print the training log every one hundred iterations
            self.print_frequency = 100

        # ==============================================================================
        # Training SRGAN model configuration
        # ==============================================================================
        if self.mode == "train_srgan":
            # Dataset address
            self.train_image_dir = "/home/lion397/data/datasets/GEMINI/Training_T4_1_2_3/train"
            self.valid_image_dir = "/home/lion397/data/datasets/GEMINI/Training_T4_1_2_3/val"


            self.image_size = 256
            self.d_image_size = 96
            self.stn_image_size = self.d_image_size # It will resize the image
            self.batch_size = 1
            self.num_workers = 1 # more than 4 is slower

            # Incremental training and migration training
            self.resume = False
            self.strict = False
            self.start_epoch = 0
            self.resume_d_weight = f"results/{exp_name}/d-last.pth"
            self.resume_g_weight = f"results/{exp_name}/g-last.pth"

            # Total num epochs
            #epochs = sys.maxsize # Very large number
            self.epochs = 8000 # Very large number

            # Loss function weight

            self.pixel_weight = 1.0
            self.content_weight = 1.0
            self.adversarial_weight = 0.004

            self.adversarial_weight_step_size = 400
            self.adversarial_weight_step_rate = 2

            self.similaity_weight = 0.0

            self.lambda_smooth = 0.0

            # Adam optimizer parameter for Discriminator
            self.d_model_lr = 1e-4 # Defalut 1e-4
            self.g_model_lr = 1e-4
            self.d_model_betas = (0.9, 0.999)
            self.g_model_betas = (0.9, 0.999)

            # MultiStepLR scheduler parameter for SRGAN
            self.d_scheduler_step_size = 400
            self.g_scheduler_step_size = 400

            self.d_scheduler_gamma = 0.1
            self.g_scheduler_gamma = 0.1

            # Print the training log every one hundred iterations
            self.print_frequency = 100

        # ==============================================================================
        # Verify configuration
        # ==============================================================================
        if mode == "valid":
            # Test data address
            if 0:
                self.lr_dir = f"/home/lion397/data/datasets/GEMINI/Training_IR_SIM_220531/val/IR_LOW"
                self.rgb_dir = f"/home/lion397/data/datasets/GEMINI/Training_IR_SIM_220531/val/RGB"
                self.sr_dir = f"results/test/{exp_name}"
                self.hr_dir = f"/home/lion397/data/datasets/GEMINI/Training_IR_SIM_220531/val/IR_HIGH"
            elif 0:
                self.lr_dir = f"/home/lion397/data/datasets/GEMINI/Training_220315/val/IR_LOW"
                self.rgb_dir = f"/home/lion397/data/datasets/GEMINI/Training_220315/val/RGB"
                self.sr_dir = f"results/test/{exp_name}"
                self.hr_dir = f"/home/lion397/data/datasets/GEMINI/Training_220315/val/IR_HIGH"
            else:
                self.lr_dir = f"/home/lion397/data/datasets/GEMINI/Training_All_221201/val/IR_LOW"
                self.hr_dir = f"/home/lion397/data/datasets/GEMINI/Training_All_221201/val/IR_HIGH"

            #model_path = f"results/{exp_name}/g-best.pth"
            #model_path = f"results/{exp_name}/srresnet-ImageNet-2df2c5f9.pth"
            self.model_path = f"results/{exp_name}/g-best.pth"