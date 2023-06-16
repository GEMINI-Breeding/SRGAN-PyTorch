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

# ==============================================================================
# File description: Realize the model definition function.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

from models.networks import ResnetGenerator
from models.spatial_transformer_net import AffineSTN
from models.dcn import DeformableConv2d

__all__ = [
    "ResidualConvBlock",
    "Discriminator", "Generator",
    "ContentLoss"
]

@torch.jit.script    
def matchTemplateTorchCore(img_tensor, template_tensor):
    result1 = torch.nn.functional.conv2d(img_tensor, template_tensor, bias=None, stride=1, padding=0)
    result2 = torch.sqrt(torch.sum(template_tensor**2) * torch.nn.functional.conv2d(img_tensor**2, torch.ones_like(template_tensor), bias=None, stride=1, padding=0))

    return (result1/result2).squeeze(0).squeeze(0)
    #return (result1).squeeze(0).squeeze(0)
   
def matchTemplateThetaBatch(background, template):
    batch_size = background.shape[0]
    theta = torch.zeros(batch_size, 6).to(background.device)

    for i in range(batch_size):
        template_i = template[i].unsqueeze(0)
        # Stretch template from 0 to 1
        template_i = (template_i - torch.min(template_i)) / (torch.max(template_i) - torch.min(template_i))
        
        background_i = F.interpolate(background[i].unsqueeze(0), size=(template_i.shape[-2], template_i.shape[-1]), mode='bilinear', align_corners=False)
        # Add padding to low img
        x_offset = background.shape[-2] // 4
        y_offset = background.shape[-1] // 4
        background_i = F.pad(background_i, (x_offset, x_offset, y_offset, y_offset), mode='replicate')
        res = matchTemplateTorchCore(background_i, template_i)
        result_max = torch.max(res)
        result_max_loc = torch.argmax(res)
        result_max_loc_x = result_max_loc % res.shape[0] 
        result_max_loc_y = result_max_loc // res.shape[1] 
        # print(result_max_loc_x, result_max_loc_y)
        # If too much transformation

        x_t = -2*(result_max_loc_x-x_offset)/ template.shape[-2]
        y_t = -2*(result_max_loc_y-y_offset) / template_i.shape[-1]
        if abs(x_t) + abs(y_t) > 0.8:
            # print("# Reset to 0")
            x_t = y_t = 0 # Reset to 0
    
        # Make theta matrix from max location
        theta[i] = torch.tensor([1., 0., x_t, 0., 1., y_t])
        theta[i] = theta[i].unsqueeze(0).repeat(background_i.shape[0],1,1)

    return theta

class ResidualConvBlock(nn.Module):
    """Implements residual conv function.

    Args:
        channels (int): Number of channels in the input image.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)
        out = torch.add(out, identity)

        return out


class Discriminator(nn.Module):
    def __init__(self,image_size=96) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96 => Changed to gray scale (1) x 96 x 96
            #nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.downsample_img = nn.UpsamplingBilinear2d(size=(image_size,image_size))

        self.classifier = nn.Sequential(
            nn.Linear(512 * image_size//16 * image_size//16, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        oux = self.downsample_img(x)
        out = self.features(oux)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self,stn_image_size=96,debug=False,export_onnx=False) -> None:
        super(Generator, self).__init__()
        self.export_onnx = export_onnx
        self.stn_image_size = stn_image_size
        self.debug = debug
        # First conv layer.
        self.conv_block1_ir = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1)), # for IR image (1, 64, 3)
            nn.PReLU(),
        )

        self.conv_block1_cycleGAN = nn.Sequential(
            nn.Conv2d(5, 64, (3, 3), (1, 1), (1, 1)), # for IR image (1, 64, 3)
            nn.PReLU(),
        )

        self.conv_block1_rgb = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)), # For RGB Image (3, 64, 3)
            nn.PReLU(),
        )

        self.rgb2ir = ResnetGenerator(3, 1, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        self.ir2rgb = ResnetGenerator(1, 3, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)     

        # Features trunk blocks.
        trunk = []
        for _ in range(16):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)

        trunk_ir = []
        for _ in range(16):
            trunk_ir.append(ResidualConvBlock(64))
        self.trunk_ir = nn.Sequential(*trunk_ir)

        trunk_rgb = []
        for _ in range(16):
            trunk_rgb.append(ResidualConvBlock(64))
        self.trunk_rgb = nn.Sequential(*trunk_rgb)

        resBlock = []
        for _ in range(1):
            resBlock.append(ResidualConvBlock(64))
        self.resBlock = nn.Sequential(*resBlock)

        # 1x1 conv layer.
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(128, 64, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )

        # Second conv layer.
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64),
        )

        self.conv_block2_ir = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64),
        )

        self.conv_block2_rgb = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64),
        )

        # Upscale conv block.
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        # Upsampling image
        if 0:
            self.upsampling_img = nn.Sequential(
                nn.Conv2d(1, 4, (3, 3), (1, 1), (1, 1)),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(1, 4, (3, 3), (1, 1), (1, 1)),
                nn.PixelShuffle(2),
                nn.PReLU(),
            )
        else:
            self.upsampling_img = nn.UpsamplingBilinear2d(scale_factor=4)


        # Output layer.
        if 0:
            self.conv_block3 = nn.Conv2d(64, 1, (9, 9), (1, 1), (4, 4))
        else:
            self.conv_block3 = nn.Sequential(
                nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0)),
                nn.PReLU(),
                nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
                nn.PReLU(),
                nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                nn.PReLU(),
                nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                nn.PReLU(),
                nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                nn.PReLU(),
                nn.Conv2d(32, 1, (1, 1), (1, 1), (0, 0))
            )
        # Initialize neural network weights.
        self._initialize_weights()

    def forward(self, x, y: Tensor) -> Tensor:
        return self._forward_impl(x, y) # x: IR, y: RGB

    # Support torch.script function.
    def _forward_impl(self, x, y: Tensor) -> Tensor:
        if self.debug:
            debug = []

        # RGB 2 IR
        out_rgb2ir = self.rgb2ir(y)
        if self.debug:
            debug.append(out_rgb2ir)
        # For cycle GAN loss calculation
        self.out_rgb2ir = out_rgb2ir
        self.out_rgb2ir2rgb = self.ir2rgb(out_rgb2ir)

        # IR 2 RGB
        out_ir2rgb = self.ir2rgb(x)
        if self.debug:
            debug.append(out_ir2rgb)
        # For cycle GAN loss calculation
        self.out_ir2rgb = out_ir2rgb
        self.out_ir2rgb2ir = self.rgb2ir(out_ir2rgb)

        
        # Template matching
        x_stn = F.interpolate(x, size=(self.stn_image_size, self.stn_image_size), mode='bilinear', align_corners=False)
        out_rgb2ir_stn = F.interpolate(out_rgb2ir, size=(self.stn_image_size, self.stn_image_size), mode='bilinear', align_corners=False)
        theta = matchTemplateThetaBatch(x_stn, out_rgb2ir_stn)

        if self.export_onnx == False:
            resampling_grid = F.affine_grid(theta.view(-1, 2, 3), out_rgb2ir.size())
            out_rgb2ir_aligned = F.grid_sample(out_rgb2ir, resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False) # 'zeros', 'border', or 'reflection'
            self.out_rgb2ir_aligned = out_rgb2ir_aligned # For loss calculation
            resampling_grid = F.affine_grid(theta.view(-1, 2, 3), y.size())
            y_aligned = F.grid_sample(y, resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False) # 'zeros', 'border', or 'reflection'
        else:
            #aten::affine_grid_generator Not yet supported. See https://pytorch.org/docs/stable/onnx_supported_aten_ops.html
            out_rgb2ir_aligned = out_rgb2ir # For loss calculation
            y_aligned = y
            
        self.y_aligned = y_aligned
        
        if self.debug:
            debug.append(self.out_rgb2ir_aligned)

        # RGB
        out_ir_1 = self.upsampling_img(x) # Pass to before last conv block
        out_rgb_ir = torch.cat((out_ir_1, out_rgb2ir_aligned, y_aligned), 1) 

        out_rgb_ir_1 = self.conv_block1_cycleGAN(out_rgb_ir)
        out_rgb_ir_2 = self.trunk(out_rgb_ir_1) 
        out_rgb_ir_3 = self.conv_block2_ir(out_rgb_ir_2)
        out_rgb_ir_4 = torch.add(out_rgb_ir_1, out_rgb_ir_3)

        if self.debug:
            debug.append(out_rgb_ir_4)

        # Final conv
        out = self.conv_block3(out_rgb_ir_4)
        if self.debug:
            return out, debug
        else:
            return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def calc_feature_corr(self, x, y: Tensor) -> Tensor:
        # Flatten features
        x_intensity = torch.mean(torch.abs(x),dim=1)
        y_intensity = torch.mean(torch.abs(y),dim=1)

        x_flat = x.view(x_intensity.size(0), -1)
        y_flat = y.view(y_intensity.size(0), -1)

        # concat two features
        xy = torch.cat((x_flat, y_flat), 0)

        # Calc corr
        corr = torch.corrcoef(xy)[0][1]
        return corr
                
class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19 = models.vgg19(pretrained=True).eval()
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        # Standardized operations
        sr = sr.sub(self.mean).div(self.std)
        hr = hr.sub(self.mean).div(self.std)

        # Find the feature map difference between the two images
        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss
