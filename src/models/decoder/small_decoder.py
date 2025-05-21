import torch.nn as nn

from src.registry.registry import GLOBAL_REGISTRY

@GLOBAL_REGISTRY.register(category='decoder', name='SmallDecoder')
class SmallDecoder(nn.Module):
    def __init__(self, latent_channels, out_channels):
        super().__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels=self.latent_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x