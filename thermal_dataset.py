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

# get current file's directory
__dir__ = os.path.dirname(os.path.abspath(__file__))
import sys
# Append root directory
sys.path.append(os.path.abspath(os.path.join(__dir__)))

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
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

def calc_temp(image, temp_range):
    if 0:
        cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX).astype(np.float64)
    else:
        span = np.max(image) - np.min(image)
        image = (image - np.min(image)) / span
    # Check image type
    res = image * (temp_range[1] - temp_range[0]) + temp_range[0]


    return res

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
        self.low_filenames = [os.path.join(low_dataroot, x) for x in os.listdir(low_dataroot) if x.split('.')[-1] in ["jpg","png","tiff"]]
        self.low_filenames.sort()
        self.high_filenames = [os.path.join(high_dataroot, x) for x in os.listdir(high_dataroot) if x.split('.')[-1] in ["jpg","png","tiff"]]
        self.high_filenames.sort()

        self.rgb_filenames = [os.path.join(rgb_dataroot, x) for x in os.listdir(rgb_dataroot) if x.split('.')[-1] in ["jpg","png","bmp"]]
        self.rgb_filenames.sort()

        self.mode = mode

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

        try:
            # Read a batch of image data
            # FLIR
            if self.low_filenames[batch_index].split('.')[-1] == "tiff":
                lr_celcious_image = cv2.imread(self.low_filenames[batch_index],-1)  
                # Convert to Celsius
                lr_celcious_image = lr_celcious_image / 100 - 273.15
            else:
                lr_celcious_image = cv2.imread(self.low_filenames[batch_index])  # FLIR

            # VarioCAM
            if self.high_filenames[batch_index].split('.')[-1] == "tiff":
                hr_celcious_image = cv2.imread(self.high_filenames[batch_index],-1)
                # Convert to Celsius
                hr_celcious_image = hr_celcious_image / 100 - 273.15
            else:
                hr_celcious_image = cv2.imread(self.high_filenames[batch_index])

            rgb_image = cv2.imread(self.rgb_filenames[batch_index])

            # Shape check 1
            h_frac = hr_celcious_image.shape[0] % self.upscale_factor
            w_frac = hr_celcious_image.shape[1] % self.upscale_factor
            if h_frac == 0 and w_frac == 0:
                pass
            else:
                hr_celcious_image = cv2.resize(hr_celcious_image, dsize=(hr_celcious_image.shape[1]-w_frac,hr_celcious_image.shape[0]-h_frac))
            
            rgb_image = cv2.resize(rgb_image, dsize=(hr_celcious_image.shape[1],hr_celcious_image.shape[0]))
            

            # Shape check
            if hr_celcious_image.shape[0] // self.upscale_factor == lr_celcious_image.shape[0] and hr_celcious_image.shape[1] // self.upscale_factor == lr_celcious_image.shape[1]:
                pass
            else:
               lr_celcious_image = cv2.resize(lr_celcious_image,dsize=(hr_celcious_image.shape[1]//self.upscale_factor,hr_celcious_image.shape[0]//self.upscale_factor))
            
            
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)         
            print(f"Error reading {self.low_filenames[batch_index]}")

        # Update self variables if everything is OK
        if lr_celcious_image is not None and hr_celcious_image is not None and rgb_image is not None:
            self.lr_celcious_image = lr_celcious_image
            self.rgb_image = rgb_image
            self.hr_celcious_image = hr_celcious_image
        
        return self.lr_celcious_image, self.rgb_image, self.hr_celcious_image



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
        
        # Calculate temperature range
        norm_info_dict = {}
        norm_info_dict["lr"] = [lr_image.min(), lr_image.max()]
        norm_info_dict["hr"] = [hr_image.min(), hr_image.max()]

        # Convert celcius to uint8
        lr_image = cv2.normalize(lr_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hr_image = cv2.normalize(hr_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Convert to Gray if the image is RGB
        if len(lr_image.shape) == 3:
            lr_image = cv2.cvtColor(lr_image,cv2.COLOR_BGR2GRAY)
        if len(hr_image.shape) == 3:
            hr_image = cv2.cvtColor(hr_image,cv2.COLOR_BGR2GRAY)

        # Transform image
        hr_image = self.hr_transforms(hr_image)
        lr_image = self.lr_transforms(lr_image)
        rgb_image = self.hr_transforms(rgb_image)
        
        if self.mode == "train":
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

        return lr_tensor, rgb_tensor, hr_tensor, norm_info_dict

    def __len__(self) -> int:
        return len(self.low_filenames)


if __name__ == "__main__":

    upscale_factor = 4
    #sample_dataset = ThermalImageDataset(dataroot="/home/lion397/data/datasets/GEMINI/TLinear_All_2023_06_01/train",
    sample_dataset = ThermalImageDataset(dataroot="/home/lion397/data/datasets/GEMINI/Training_T4_1_2_3/train",
    #sample_dataset = ThermalImageDataset(dataroot="/home/GEMINI/Dataset_processing/Davis_Legumes/2022-07-06/Thermal_Matched_old",
                                        image_size=96, upscale_factor=upscale_factor, mode="train")

    i = 0
    while True:
    #for i in range(len(sample_dataset.low_filenames)):
        print(f"[{i}]{sample_dataset.low_filenames[i]}")
        i = np.clip(i,0,len(sample_dataset.low_filenames)-1)
        (low_img, rgb_img, high_img) = sample_dataset.getImage(i)
        # Convert celcius to uint8ddd
        low_img_vis = cv2.normalize(low_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        high_img_vis = cv2.normalize(high_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        (low_tensor, rgb_tensor, high_tensor, norm_info_dict) = sample_dataset[i]

        if 1:
            disp_img = cv2.hconcat((cv2.resize(low_img_vis,dsize=(0,0),fx=upscale_factor, fy=upscale_factor),high_img_vis))
            # Check the image is color or gray
            if len(disp_img.shape) == 2:
                disp_img = cv2.cvtColor(disp_img,cv2.COLOR_GRAY2BGR)
            
            disp_img = cv2.hconcat((rgb_img,disp_img))
            disp_img = cv2.resize(disp_img,dsize=(0,0),fx=1/4, fy=1/4)
            cv2.imshow("disp_img",disp_img)

            key = cv2.waitKey(-1)
            if key == ord("q"):
                break
            elif key == ord("a"):
                i -= 1
            elif key == ord("d"):
                i += 1
            elif key == ord("s"):
                i -= 10
            elif key == ord("w"):
                i += 10
                # os.sys.exit(0)