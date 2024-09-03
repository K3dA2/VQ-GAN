import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import Encoder, Decoder, ResNet

class Discriminator(nn.Module):
    def __init__(self, width=64, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResNet(width, width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResNet(width, width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 3, kernel_size=3, stride = 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width * 3, width * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNet(width * 3, width),
            nn.ReLU(),
            ResNet(width, 1)
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorLoss():
    def __init__(self, device = "cpu",gan_mode='lsgan'):
        self.discriminator = Discriminator().to(device)
        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN mode {gan_mode} not implemented")

    def calculate_loss(self, real, fake):
        real_out = self.discriminator(real)
        fake_out = self.discriminator(fake)

        # Target labels
        ones = torch.ones_like(real_out).to(real.device)
        zeros = torch.zeros_like(fake_out).to(fake.device)

        # Calculate losses
        r_loss = self.criterion(real_out, ones)
        f_loss = self.criterion(fake_out, zeros)

        # Combine losses
        gan_loss = (r_loss + f_loss) * 0.5  # Average of real and fake loss

        return gan_loss, fake_out
