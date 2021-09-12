
'''
    Rico Zhu - September 8th, 2021 @ Duke University
    
    Pytorch implementation of a DCGAN, Progressive GAN, and a StyleGAN
    DCGAN inspiration: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

'''

import os
import random

import numpy as np
import matplotlib as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

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

def train(epochs, dataloader, b1, b2, sample_interval=400):

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Size of each image dimension, dimensionality of the latent space, size of convolution, number of image channels
    generator = Generator(32, 100, 128, 1)
    discriminator = Discrimator(32, 128, 1)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Generator optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(b1, b2))
    # Discrimnator optimizer
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(b1, b2))

    Tensor = torch.FloatTensor

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            real_imgs = Variable(imgs.type(Tensor))

            # Train generator
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train discriminator
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

# Create output directory for generated images
os.makedirs("images", exist_ok=True)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=64,
    shuffle=True,
)

train(200, dataloader, 0.5, 0.999)