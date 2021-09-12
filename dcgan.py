
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

# Function to initialize a given weight to a normal distribution
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):

    def __init__(self, img_dim, latent_dim, conv_dim, img_channels, rgb=True, **kwargs):

        super(Generator, self).__init__()
        
        # Input layer dimensions
        self.img_dim = img_dim
        self.init_dim = img_dim // 4
        self.latent_dim = latent_dim

        # Conv block dimensions
        self.conv_dim = conv_dim
        self.img_channels = img_channels
        self.conv_kernels = (3 if rgb else 1)

        # Build model layers
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

    def __init__(self, img_dim, conv_dim, img_channels, rgb=True, **kwargs):

        super(Discrimator, self).__init__()

        self.img_dim = img_dim
        self.conv_dim = conv_dim
        self.img_channels = img_channels
        self.conv_kernels = (3 if rgb else 1)

        def discriminator_block(in_filters, out_filters, conv_kernels, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, conv_kernels, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.img_channels, self.conv_dim // 8, self.conv_kernels, bn=False),
            *discriminator_block(self.conv_dim // 8, self.conv_dim // 4, self.conv_kernels),
            *discriminator_block(self.conv_dim // 4, self.conv_dim // 2, self.conv_kernels),
            *discriminator_block(self.conv_dim // 2, self.conv_dim, self.conv_kernels),
        )

        # The height and width of downsampled image
        self.ds_size = self.img_dim // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(self.conv_dim * self.ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

def train(epochs, batch_size, b1, b2, sample_interval):

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Size of each image dimension, dimensionality of the latent space, size of convolution, number of image channels
    generator = Generator(32, 100, 128, 1)
    discriminator = Discrimator(32, 128, 1)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


