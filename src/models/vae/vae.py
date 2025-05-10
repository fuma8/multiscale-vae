import torch.nn as nn

from src.registry.registry import GLOBAL_REGISTRY

@GLOBAL_REGISTRY.register(category='vae', name='base_vae')
class BaseVAE(nn.Module):
    def __init__(self, encoder, decoder, distribution):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.distribution = distribution
    
    def forward(self, x):
        z = self.encoder(x)
        posterior = self.distribution(z)
        z = posterior.sample()
        x_hat = self.decoder(z)
        return x_hat