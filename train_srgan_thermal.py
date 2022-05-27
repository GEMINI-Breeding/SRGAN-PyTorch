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
"""File description: Initialize the SRResNet model."""
import os
from tabnanny import verbose
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from thermal_dataset import ThermalImageDataset as ImageDataset
from model_thermal_rgb import Discriminator, Generator, ContentLoss

from ssim import ssim
from pytorch_similarity.torch_similarity.modules import GradientDifference2d, GradientCorrelation2d
from pytorch_similarity.torch_similarity.modules import NormalizedCrossCorrelationLoss, NormalizedCrossCorrelation
import signal
import sys
autocast_on = False

interrupted = False
def handler(signum, _):
    print(f'Application is terminated by {signal.Signals(signum).name}\n')
    global interrupted
    interrupted = True

def main():

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGQUIT, handler)
    signal.signal(signal.SIGABRT, handler)
    signal.signal(signal.SIGTERM, handler)

    print("Load train dataset and valid dataset...")
    train_dataloader, valid_dataloader = load_dataset()
    print("Load train dataset and valid dataset successfully.")

    print("Build SRGAN model...")
    discriminator, generator = build_model()
    print("Build SRGAN model successfully.")

    print("Define all loss functions...")
    psnr_criterion, pixel_criterion, content_criterion, adversarial_criterion, ssim_criterion, similaity_criterion = define_loss()
    print("Define all loss functions successfully.")

    print("Define all optimizer functions...")
    d_optimizer, g_optimizer = define_optimizer(discriminator, generator)
    print("Define all optimizer functions successfully.")

    print("Define all optimizer scheduler functions...")
    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether the training weight is restored...")
    resume_checkpoint(discriminator, generator)
    print("Check whether the training weight is restored successfully.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler.
    scaler = amp.GradScaler()

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0

    print("Start train SRGAN model.")
    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch {epoch}")
        train(discriminator,
              generator,
              train_dataloader,
              psnr_criterion,
              ssim_criterion,
              similaity_criterion,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              scaler,
              writer)

        #psnr = validate_ssim(generator, valid_dataloader, psnr_criterion, epoch, writer)
        psnr = validate(generator, valid_dataloader, psnr_criterion, ssim_criterion, similaity_criterion, epoch, writer)
        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        
        if False:
            if epoch % 100 == 0:
                torch.save(discriminator.state_dict(), os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth"))
                torch.save(generator.state_dict(), os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth"))
            
        if is_best:
            torch.save(discriminator.state_dict(), os.path.join(results_dir, "d-best.pth"))
            torch.save(generator.state_dict(), os.path.join(results_dir, f"g-best.pth"))

        # Update LR
        d_scheduler.step()
        g_scheduler.step()
        writer.add_scalar("Train/g_lr", get_lr(g_optimizer), epoch)
        writer.add_scalar("Train/d_lr", get_lr(d_optimizer), epoch)

        if interrupted:
            break

    # Save the generator weight under the last Epoch in this stage
    torch.save(discriminator.state_dict(), os.path.join(results_dir, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(results_dir, "g-last.pth"))
    print("End train SRGAN model.")


def load_dataset() -> [DataLoader, DataLoader]:
    """Load super-resolution data set

     Returns:
         training data set iterator, validation data set iterator

    """
    # Initialize the LMDB data set class and write the contents of the LMDB database file into memory
    train_datasets = ImageDataset(config.train_image_dir, config.image_size, config.upscale_factor, "train")
    valid_datasets = ImageDataset(config.valid_image_dir, config.image_size, config.upscale_factor, "valid")
    # Make it into a data set type supported by PyTorch
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  persistent_workers=False)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  persistent_workers=False)

    return train_dataloader, valid_dataloader


def build_model() -> nn.Module:
    """Building discriminator and generators model

    Returns:
        SRGAN model

    """
    discriminator = Discriminator().to(config.device)
    generator = Generator().to(config.device)

    return discriminator, generator


def define_loss() -> [nn.MSELoss, nn.MSELoss, ContentLoss, ssim, nn.BCEWithLogitsLoss, GradientDifference2d]:
    """Defines all loss functions

    Returns:
        PSNR loss, pixel loss, content loss, adversarial loss

    """
    psnr_criterion = nn.MSELoss().to(config.device)
    pixel_criterion = nn.MSELoss().to(config.device)
    content_criterion = ContentLoss().to(config.device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(config.device)
    ssim_criterion = ssim
    #similaity_criterion = NormalizedCrossCorrelationLoss(return_map=True).to(config.device)
    similaity_criterion = NormalizedCrossCorrelation(return_map=True).to(config.device)
    #similaity_criterion = GradientDifference2d(return_map=True).to(config.device)

    return psnr_criterion, pixel_criterion, content_criterion, adversarial_criterion, ssim_criterion, similaity_criterion


def define_optimizer(discriminator: nn.Module, generator: nn.Module) -> [optim.Adam, optim.Adam]:
    """Define all optimizer functions

    Args:
        discriminator (nn.Module): Discriminator model
        generator (nn.Module): Generator model

    Returns:
        SRGAN optimizer

    """
    d_optimizer = optim.Adam(discriminator.parameters(), config.d_model_lr, config.d_model_betas)
    g_optimizer = optim.Adam(generator.parameters(), config.g_model_lr, config.g_model_betas)

    return d_optimizer, g_optimizer


def define_scheduler(d_optimizer: optim.Adam, g_optimizer: optim.Adam) -> [lr_scheduler.StepLR, lr_scheduler.StepLR]:
    """Define learning rate scheduler

    Args:
        d_optimizer (optim.Adam): Discriminator optimizer
        g_optimizer (optim.Adam): Generator optimizer

    Returns:
        SRGAN model scheduler

    """
    d_scheduler = lr_scheduler.StepLR(d_optimizer, step_size=config.d_optimizer_step_size, gamma=config.d_optimizer_gamma, verbose=False)
    g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=config.g_optimizer_step_size, gamma=config.g_optimizer_gamma, verbose=False)

    return d_scheduler, g_scheduler


def resume_checkpoint(discriminator: nn.Module, generator: nn.Module) -> None:
    """Transfer training or recovery training

    Args:
        discriminator (nn.Module): Discriminator model
        generator (nn.Module): Generator model

    """
    if config.resume:
        if config.resume_d_weight != "":
            discriminator.load_state_dict(torch.load(config.resume_d_weight), strict=config.strict)
        if config.resume_g_weight != "":
            generator.load_state_dict(torch.load(config.resume_g_weight), strict=config.strict)


def train(discriminator,
          generator,
          train_dataloader,
          psnr_criterion,
          ssim_criterion,
          similaity_criterion,
          pixel_criterion,
          content_criterion,
          adversarial_criterion,
          d_optimizer,
          g_optimizer,
          epoch,
          scaler,
          writer) -> None:
    # Calculate how many iterations there are under epoch
    batches = len(train_dataloader)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_hr_probabilities = AverageMeter("D(HR)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, adversarial_losses,
                              d_hr_probabilities, d_sr_probabilities,
                              psnres],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put all model in train mode.
    discriminator.train()
    generator.train()

    end = time.time()
    for index, (lr, rgb, hr) in enumerate(train_dataloader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        # Send data to designated device
        lr = lr.to(config.device, non_blocking=True)
        hr = hr.to(config.device, non_blocking=True)
        rgb = rgb.to(config.device, non_blocking=True)

        # Set the real sample label to 1, and the false sample label to 0
        real_label = torch.full([lr.size(0), 1], 1.0, dtype=lr.dtype, device=config.device)
        fake_label = torch.full([lr.size(0), 1], 0.0, dtype=lr.dtype, device=config.device)

        # Use generators to create super-resolution images
        sr = generator(lr, rgb)

        # Start training discriminator
        # At this stage, the discriminator needs to require a derivative gradient
        for p in discriminator.parameters():
            p.requires_grad = True

        # Initialize the discriminator optimizer gradient
        d_optimizer.zero_grad()

        # Calculate the loss of the discriminator on the high-resolution image
        if autocast_on:
            with amp.autocast():
                hr_output = discriminator(hr)
                d_loss_hr = adversarial_criterion(hr_output, real_label)
        else:
            hr_output = discriminator(hr)
            d_loss_hr = adversarial_criterion(hr_output, real_label)
        # Gradient zoom
        scaler.scale(d_loss_hr).backward()

        # Calculate the loss of the discriminator on the super-resolution image.
        if autocast_on:
            with amp.autocast():
                sr_output = discriminator(sr.detach())
                d_loss_sr = adversarial_criterion(sr_output, fake_label)
        else:
            sr_output = discriminator(sr.detach())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
        # Gradient zoom
        scaler.scale(d_loss_sr).backward()
        # Update discriminator parameters
        scaler.step(d_optimizer)
        scaler.update()

        # Count discriminator total loss
        d_loss = d_loss_hr + d_loss_sr
        # End training discriminator

        # Start training generator
        # At this stage, the discriminator no needs to require a derivative gradient
        for p in discriminator.parameters():
            p.requires_grad = False

        # Initialize the generator optimizer gradient
        g_optimizer.zero_grad()

        # Calculate the loss of the generator on the super-resolution image
        if autocast_on:
            with amp.autocast():
                output = discriminator(sr)
                pixel_loss = config.pixel_weight * pixel_criterion(sr, hr.detach())
                content_loss = config.content_weight * content_criterion(sr, hr.detach())
                adversarial_loss = config.adversarial_weight * adversarial_criterion(output, real_label) 
        else:
            output = discriminator(sr)
            pixel_loss =  config.pixel_weight * pixel_criterion(sr, hr.detach())
            content_loss = config.content_weight * content_criterion(sr, hr.detach())
            adversarial_loss = config.adversarial_weight * adversarial_criterion(output, real_label)

        
        similaity_val, _ = similaity_criterion(sr, hr.detach())
        if 1:
            #similaity_loss = config.similaity_weight * (1.0 - similaity_val) # Loss function for NCC
            similaity_loss = config.similaity_weight * (-torch.log10(similaity_val))
        else:
            similaity_loss = config.similaity_weight * similaity_val # Loss function for Gradient Differnce

        # Count discriminator total loss
        g_loss = (pixel_loss
                  + similaity_loss
                  + content_loss
                  + adversarial_loss)

        # Gradient zoom
        scaler.scale(g_loss).backward()
        # Update generator parameters
        scaler.step(g_optimizer)
        scaler.update()

        # End training generator

        # Calculate the scores of the two images on the discriminator
        d_hr_probability = torch.sigmoid(torch.mean(hr_output))
        d_sr_probability = torch.sigmoid(torch.mean(sr_output))

        # measure accuracy and record loss
        psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_hr_probabilities.update(d_hr_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))
        psnres.update(psnr.item(), lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        iters = index + epoch * batches + 1
        writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
        writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
        writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
        #writer.add_scalar("Train/SSIM_Loss", ssim_loss.item(), iters)
        writer.add_scalar("Train/similaity_Loss", similaity_loss.item(), iters)
        writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
        writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
        writer.add_scalar("Train/D(HR)_Probability", d_hr_probability.item(), iters)
        writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
        if index % config.print_frequency == 0 and index != 0:
            progress.display(index)


def validate(model, valid_dataloader, psnr_criterion, ssim_criterion, similaity_criterion, epoch, writer) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimres = AverageMeter("SSIM", ":4.2f")
    similaityres = AverageMeter("Similaity", ":4.2f")
    progress = ProgressMeter(len(valid_dataloader), [batch_time, psnres], prefix="Valid: ")

    # Put the generator in verification mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for index, (lr, rgb, hr) in enumerate(valid_dataloader):
            lr = lr.to(config.device, non_blocking=True)
            hr = hr.to(config.device, non_blocking=True)
            rgb = rgb.to(config.device, non_blocking=True)

            if autocast_on:
                # Mixed precision
                with amp.autocast():
                    sr = model(lr, rgb)
            else:
                sr = model(lr, rgb)

            # measure accuracy and record loss
            psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
            psnres.update(psnr.item(), hr.size(0))

            if autocast_on:
                # Mixed precision
                with amp.autocast():
                    ssim_val = ssim_criterion(sr, hr)
                    similaity_val, _ = similaity_criterion(sr, hr.detach())
            else:
                ssim_val = ssim_criterion(sr, hr)
                similaity_val, _ = similaity_criterion(sr, hr.detach())

            ssimres.update(ssim_val.item(), hr.size(0))
            similaityres.update(similaity_val.item(), hr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % config.print_frequency == 0:
                progress.display(index)


        # Tensorboard
        writer.add_scalar("Valid/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar("Valid/SSIM", ssimres.avg, epoch + 1)
        writer.add_scalar("Valid/Similaity", similaityres.avg, epoch + 1)

        # Print evaluation indicators.
        print(f"* PSNR: {psnres.avg:4.2f}")
        print(f"* SSIM: {ssimres.avg:4.2f}")
        print(f"* Similaity: {similaityres.avg:4.2f}")
        
        if epoch % 10 == 0:
            # Test Image
            sample_dataset = ImageDataset(dataroot="/home/lion397/data/datasets/GEMINI/Training_220315/val/",
                                            image_size=96, upscale_factor=4, mode="train")
            (low_img, rgb_img, high_img) = sample_dataset.getImage(10)

            with amp.autocast():
                
                low_img = torch.FloatTensor(low_img)[None, None,:, :]/255.0
                rgb_img = rgb_img[:, :, [2, 1, 0]] # swap channel from RGB to BGR
                rgb_img = torch.FloatTensor(rgb_img).permute(2,0,1).unsqueeze(0)/255.0
                
                high_img = torch.FloatTensor(high_img)[None, None,:, :]/255.0

                lr = low_img.to(config.device, non_blocking=True)
                rgb = rgb_img.to(config.device, non_blocking=True)
                high_img = high_img.to(config.device, non_blocking=True)
            
                sr = model(lr, rgb)

            if epoch == 0:
                # Write once
                writer.add_image("Valid/Input_IR",lr.squeeze(0),epoch + 1 )
                writer.add_image("Valid/Input_RGB",rgb.squeeze(0),epoch + 1 )
                writer.add_image("Valid/GroundTruth",high_img.squeeze(0),epoch + 1 )
            # Write everytime
            writer.add_image("Valid/Output",sr.squeeze(0),epoch + 1 )

    return psnres.avg



def validate_ssim(model, valid_dataloader, psnr_criterion, epoch, writer) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("SSIM", ":4.2f")
    progress = ProgressMeter(len(valid_dataloader), [batch_time, psnres], prefix="Valid: ")

    # Put the generator in verification mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for index, (lr, rgb, hr) in enumerate(valid_dataloader):
            lr = lr.to(config.device, non_blocking=True)
            hr = hr.to(config.device, non_blocking=True)

            if autocast_on:
                # Mixed precision
                with amp.autocast():
                    sr = model(lr, rgb)
            else:
                sr = model(lr, rgb)

            # measure accuracy and record loss
            # psnr = psnr_criterion(sr, hr)
            psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
            psnres.update(psnr.item(), hr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % config.print_frequency == 0:
                progress.display(index)

        writer.add_scalar("Valid/SSIM", psnres.avg, epoch + 1)
        # Print evaluation indicators.
        print(f"* SSIM: {psnres.avg:4.2f}.\n")

    return psnres.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    main()

