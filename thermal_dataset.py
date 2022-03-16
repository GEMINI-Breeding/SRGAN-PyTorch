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

import imgproc
import cv2
import math

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

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        
        self.DEBUG = True
        low_dataroot = os.path.join(dataroot,"IR_LOW")
        high_dataroot = os.path.join(dataroot,"IR_HIGH")

        super(ThermalImageDataset, self).__init__()
        self.low_filenames = [os.path.join(low_dataroot, x) for x in os.listdir(low_dataroot) if x.split('.')[-1] in ["jpg","png","bmp"]]
        self.low_filenames.sort()
        self.high_filenames = [os.path.join(high_dataroot, x) for x in os.listdir(high_dataroot) if x.split('.')[-1] in ["jpg","png","bmp"]]
        self.high_filenames.sort()

        if mode == "train" and False:
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(image_size),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(0.5),
            ])
        else:
            self.hr_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(image_size)
            ])
            

        if 0:
            self.lr_transforms = transforms.Resize(image_size // upscale_factor, interpolation=IMode.BICUBIC, antialias=True)
        else:
            self.lr_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(image_size // upscale_factor)
            ])
            # There is no way to random crop both lr and hr image in same position

        


    def getImage(self, batch_index: int):
        # Read a batch of image data
        self.lr_image = cv2.imread(self.low_filenames[batch_index])  # FLIR
        self.hr_image = cv2.imread(self.high_filenames[batch_index]) # VarioCAM

        if 0:
            self.lr_image = cv2.cvtColor(self.lr_image,cv2.COLOR_BGR2GRAY)
            self.hr_image = cv2.cvtColor(self.hr_image,cv2.COLOR_BGR2GRAY)

        cv2.normalize(self.lr_image, self.lr_image, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(self.hr_image, self.hr_image, 0, 255, cv2.NORM_MINMAX)

        
        return self.lr_image, self.hr_image



    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        
        (lr_image, hr_image) = self.getImage(batch_index)

        # Transform image
        hr_image = self.hr_transforms(hr_image)
        lr_image = self.lr_transforms(lr_image)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.low_filenames)


if __name__ == "__main__":

    sample_dataset = ThermalImageDataset(dataroot="data/Training_220315/train",
                                        image_size=96, upscale_factor=4, mode="train")

    for i in range(len(sample_dataset.low_filenames)):
        (low_img, high_img) = sample_dataset.getImage(i)
        (low_tensor, high_tensor) = sample_dataset[i]

