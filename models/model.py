"""
Heesup Yun & Pranv Raja 
Implementation of 
"
Arar, M., Ginger, Y., Danon, D., Bermano, A. H., & Cohen-Or, D. (2020). 
Unsupervised multi-modal image registration via geometry preserving image-to-image translation. 
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 13410-13419).
"
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

# import models.stn as stn
from .spatial_transformer_net import AffineSTN
from . import networks
from .base_model import BaseModel

from .ssim import ssim

class Net(BaseModel):
    def __init__(self, opt):
        super(Net, self).__init__(opt)

        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids

        # Define Spatial Transformer Network
        self.netSTN = AffineSTN(opt.input_nc, opt.output_nc, opt.img_height, opt.img_width, opt.init_type).to(self.device)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG_type, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # Define discriminator
        if self.isTrain: 
            self.netD = networks.define_D(opt.output_nc + opt.input_nc, opt.ndf, opt.netD_type,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        # Define loss functions and optimiziers
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            if 0:
                self.criterionL = torch.nn.L1Loss()
            else:
                self.criterionL = torch.nn.MSELoss()
    

        self.ssim = ssim

        if self.isTrain:
            self.model_names = ["STN","G","D"]
        else:
            self.model_names = ["STN","G"]

        if self.opt.continue_train or self.isTrain == False:
            self.load_networks(epoch=self.opt.load_iter)


    def forward(self, real_A, real_B):
        self.real_A = real_A
        self.real_B = real_B

        self.fake_B = self.netG(real_A)
        wraped_images = self.netSTN(real_A, real_B, apply_on=[real_A, self.fake_B])
        self.stn_reg_term = torch.mean(torch.abs(self.netSTN.dtheta))
        self.registered_real_A = wraped_images[0]
        self.registered_fake_B = wraped_images[1]
        with torch.no_grad():
            self.deformation_field_A_to_B = self.netSTN.get_grid(real_A.size())

    def backward(self):
        # Backward D
        self.set_requires_grad([self.netG, self.netSTN], False)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB.detach())
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Fake considering STN
        fake_AB_1 = torch.cat((self.real_A, self.registered_fake_B), 1)
        pred_fake_1 = self.netD(fake_AB_1.detach())
        fake_AB_2 = torch.cat((self.registered_real_A, self.fake_B), 1)
        pred_fake_2 = self.netD(fake_AB_2.detach())
        self.loss_D_fake = 0.5 * (self.criterionGAN(pred_fake_1, False) + self.criterionGAN(pred_fake_2, False))

        # combine loss and calculate gradients
        self.loss_D = 0.5 * self.opt.lambda_GAN * (self.loss_D_real + self.loss_D_fake)
        self.loss_D.backward()

        self.optimizer_D.step()  # update D_A and D_B's weights
        self.set_requires_grad([self.netG, self.netSTN], True)


        # Optimize generator
        self.set_requires_grad([self.netD], False)
        self.optimizer_STN.zero_grad()
        self.optimizer_G.zero_grad()  # clear previous gradient history of G_A and G_B

        # L1 Loss or L2 Loss
        self.loss_L = self.opt.lambda_recon * self.criterionL(self.registered_fake_B, self.real_B)

        # GAN loss 
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_GAN = self.opt.lambda_GAN * self.criterionGAN(pred_fake, True)
        
        self.loss_smoothness = self.opt.lambda_smooth * self.stn_reg_term
        self.ssim_value = self.ssim(self.registered_real_A, self.real_B)
        self.ssim_Loss = (1.0 - self.ssim_value)*10 
        self.loss_G =  self.loss_L + self.loss_GAN + self.loss_smoothness + self.ssim_Loss
        self.loss_G.backward()


        self.optimizer_STN.step()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD], True)

