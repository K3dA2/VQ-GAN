import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest

class ResNet(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.network = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.network(x)
        return torch.add(out,self.residual_layer(x))
    

class Encoder(nn.Module):
    def __init__(self, latent_dim = 32, width = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNet(width * 3,width * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 4, width * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNet(width * 3,latent_dim)
        )
    
    def forward(self,img):
        out = self.net(img)
        
        return out

class Decoder(nn.Module):
    def __init__(self, latent_dim = 32, width = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ResNet(latent_dim,width * 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            ResNet(width * 3,width * 3),
            nn.ReLU(),
            nn.Conv2d(width * 3, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(width * 3, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )
        self.last_layer = nn.Conv2d(width, 3, kernel_size=3, padding=1)
    
    def forward(self,img):
        out = self.net(img)
        out = self.last_layer(out)
        return out
    
class TestDecoder(unittest.TestCase):
    def test_decoder_output_shape(self):
        # Create a dummy input image with latent representation size
        img = torch.randn(2, 32, 16, 16)
        
        # Initialize the decoder
        decoder = Decoder(latent_dim=32)
        
        # Pass the image through the decoder
        out = decoder(img)
        
        print(out.shape)
        # Check the output shape
        self.assertEqual(out.shape, (2, 3, 64, 64), f"Output shape mismatch: expected (2, 32, 32, 32), got {out.shape}")


if __name__ == '__main__':
    unittest.main()