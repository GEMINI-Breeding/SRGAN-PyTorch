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
"""Realize the function of dataset preparation."""
import io
import os

import lmdb
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode
import torchvision.transforms.functional as TF

import imgproc
import cv2
import math
import random
__all__ = ["ThermalImageDataset", "LMDBDataset"]


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R


class ThermalImageDataset(Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.

    Args:
        dataroot         (str): Training data set address
        image_size       (int): High resolution image size
        upscale_factor   (int): Image magnification
        mode             (str): Data set loading method, the training data set is for data enhancement,
                             and the verification data set is not for data enhancement

    """

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str, random_crop=True) -> None:
        
        self.DEBUG = True
        low_dataroot = os.path.join(dataroot,"IR_LOW")
        high_dataroot = os.path.join(dataroot,"IR_HIGH")
        rgb_dataroot = os.path.join(dataroot,"RGB")

        super(ThermalImageDataset, self).__init__()
        self.low_filenames = [os.path.join(low_dataroot, x) for x in os.listdir(low_dataroot) if x.split('.')[-1] in ["jpg","png","bmp"]]
        self.low_filenames.sort()
        self.high_filenames = [os.path.join(high_dataroot, x) for x in os.listdir(high_dataroot) if x.split('.')[-1] in ["jpg","png","bmp"]]
        self.high_filenames.sort()

        self.rgb_filenames = [os.path.join(rgb_dataroot, x) for x in os.listdir(rgb_dataroot) if x.split('.')[-1] in ["jpg","png","bmp"]]
        self.rgb_filenames.sort()

        if 0:
            if mode == "train" :
                self.hr_transforms = transforms.Compose([
                    transforms.RandomRotation(90),
                    transforms.RandomHorizontalFlip(0.5),
                ])

                self.lr_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.CenterCrop(image_size // upscale_factor)
                ])
            else:
                self.hr_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                ])
            


            if 0:
                self.lr_transforms = transforms.Resize(image_size // upscale_factor, interpolation=IMode.BICUBIC, antialias=True)
            else:
                self.lr_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.CenterCrop(image_size // upscale_factor)
                ])
                # There is no way to random crop both lr and hr image in same position
        else:
            self.hr_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                ])
            self.lr_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                ])

        self.image_size = image_size
        self.upscale_factor = upscale_factor
        self.random_crop = random_crop

        


    def getImage(self, batch_index: int):
        # Read a batch of image data
        self.lr_image = cv2.imread(self.low_filenames[batch_index])  # FLIR
        self.hr_image = cv2.imread(self.high_filenames[batch_index]) # VarioCAM
        self.rgb_image = cv2.imread(self.rgb_filenames[batch_index])
        if 1:
            self.lr_image = cv2.cvtColor(self.lr_image,cv2.COLOR_BGR2GRAY)
            self.hr_image = cv2.cvtColor(self.hr_image,cv2.COLOR_BGR2GRAY)

        cv2.normalize(self.lr_image, self.lr_image, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(self.hr_image, self.hr_image, 0, 255, cv2.NORM_MINMAX)

        
        return self.lr_image, self.rgb_image, self.hr_image



    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        
        (lr_image, rgb_image, hr_image) = self.getImage(batch_index)

        if self.random_crop:
            lr_crop_w = self.image_size // self.upscale_factor
            lr_crop_h = self.image_size // self.upscale_factor
            lr_crop_x = random.randint(0,lr_image.shape[1] - self.image_size // self.upscale_factor)
            lr_crop_y = random.randint(0,lr_image.shape[0] - self.image_size // self.upscale_factor)

            lr_image = lr_image[lr_crop_y:lr_crop_y+lr_crop_h,lr_crop_x:lr_crop_x+lr_crop_w]

            hr_crop_w = self.image_size 
            hr_crop_h = self.image_size
            hr_crop_x = int(lr_crop_x * self.upscale_factor)
            hr_crop_y = int(lr_crop_y * self.upscale_factor)

            hr_image = hr_image[hr_crop_y:hr_crop_y+hr_crop_h,hr_crop_x:hr_crop_x+hr_crop_w]

            rgb_image = rgb_image[hr_crop_y:hr_crop_y+hr_crop_h,hr_crop_x:hr_crop_x+hr_crop_w]

        # Transform image
        hr_image = self.hr_transforms(hr_image)
        lr_image = self.lr_transforms(lr_image)
        rgb_image = self.hr_transforms(rgb_image)

        if random.random() > 0.5:
            hr_image = TF.vflip(hr_image)
            lr_image  = TF.vflip(lr_image)
            rgb_image  = TF.vflip(rgb_image)

        if random.random() > 0.5:
            hr_image = TF.hflip(hr_image)
            lr_image  = TF.hflip(lr_image)
            rgb_image  = TF.hflip(rgb_image)

        if random.random() > 0.5:
            hr_image =  TF.rotate(hr_image,90)
            lr_image  = TF.rotate(lr_image,90)
            rgb_image  = TF.rotate(rgb_image,90)
       

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)
        rgb_tensor = imgproc.image2tensor(rgb_image, range_norm=False, half=False)

        return lr_tensor, rgb_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.low_filenames)


if __name__ == "__main__":

    sample_dataset = ThermalImageDataset(dataroot="/home/lion397/data/datasets/GEMINI/Training_220315/train/",
                                        image_size=96, upscale_factor=4, mode="train")

    for i in range(len(sample_dataset.low_filenames)):
        (low_img, rgb_img, high_img) = sample_dataset.getImage(i)
        (low_tensor, rgb_tensor, high_tensor) = sample_dataset[i]

