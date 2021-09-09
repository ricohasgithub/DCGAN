
'''
    Rico Zhu - September 8th, 2021 @ Duke University
    
    Pytorch implementation of a DCGAN, Progressive GAN, and a StyleGAN

'''

import os
import random

import matplotlib as plt

import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):

    def __init__(self, img_dim, latent_dim, conv_dim, rgb=True, img_channels=1, **kwargs):

        super(Generator, self).__init__()
        
        self.img_dim = img_dim
        self.init_dim = img_dim // 4
        self.latent_dim = latent_dim
        self.conv_dim = conv_dim
        self.conv_kernels = (3 if rgb else 1)
        self.img_channels = img_channels

        self.linear_in = nn.Sequential(nn.Linear(self.latent_dim, self.conv_dim * self.init_dim ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.conv_dim),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.conv_dim, self.conv_dim, self.conv_kernels, stride=1, padding=1),
            nn.BatchNorm2d(self.conv_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.conv_dim, self.conv_dim // 2, self.conv_kernels, stride=1, padding=1),
            nn.BatchNorm2d(self.conv_dim // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.conv_dim // 2, self.img_channels, self.conv_kernels, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.linear_in(x)
        out = out.view(out.shape[0], self.conv_dim, self.init_dim, self.init_dim)
        img = self.conv_blocks(out)
        return img

class Discrimator(nn.Module):

    def __init__(self, **kwargs):
        super(Discrimator, self).__init__()
        pass
