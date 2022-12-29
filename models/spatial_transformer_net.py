import torch
import torch.nn.functional as F
from torch import nn

from .layers import DownBlock, get_activation


class AffineSTN(nn.Module):
    def __init__(self, nc_a, nc_b, height, width, init_func):
        super(AffineSTN, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.identity_theta = torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float).to(self.device)
        self.h, self.w = height, width
        nconvs = 5
        convs = []
        prev_nf = nc_a + nc_b
        nfs = [32, 64, 128, 256, 256]
        for nf in nfs:
            convs.append(DownBlock(prev_nf, nf, 3, 1, 1, bias=True, activation='relu',
                                   init_func=init_func, use_norm=True,
                                   use_resnet=False,
                                   skip=False,
                                   refine=False,
                                   pool=True))
            prev_nf = nf

        self.convs = nn.Sequential(*convs)
        act = get_activation(activation='relu')
        self.local = nn.Sequential(
            nn.Linear(prev_nf * (self.h // 2 ** nconvs) *
                      (self.w // 2 ** nconvs), nf, bias=True),
            act, nn.Linear(nf, 6, bias=True))

        # Start with identity transformation
        self.local[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.local[-1].bias.data.zero_()

    def get_grid(self, shape):
        """Return the predicted sampling grid that aligns img_a with img_b."""
        theta = self.theta
        resampling_grid = F.affine_grid(theta.view(-1, 2, 3), shape)
        return resampling_grid

    def forward(self, img_a, img_b, apply_on=None):

        # Get Affine transformation
        x = torch.cat([img_a, img_b], 1)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        self.dtheta = self.local(x)
        self.theta = self.dtheta + \
            self.identity_theta.unsqueeze(0).repeat(img_a.size(0), 1)

        # Wrap image wrt to the deformation field
        if apply_on is None:
            apply_on = [img_a]
        warped_images = []
        for img in apply_on:
            resampling_grid = F.affine_grid(
                self.theta.view(-1, 2, 3), img.size())
            warped_images.append(
                F.grid_sample(img, resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False))
        return warped_images

    def _calculate_regularization_term(self, theta):
        x = torch.mean(torch.abs(theta))
        return x