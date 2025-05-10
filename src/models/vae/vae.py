import torch.nn as nn
import torch.nn.functional as F

from src.registry.registry import GLOBAL_REGISTRY

@GLOBAL_REGISTRY.register(category='vae', name='base_vae')
class BaseVAE(nn.Module):
    def __init__(self, encoder, decoder, distribution):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.distribution = distribution
    
    def get_loss(self, x):
        z = self.encoder(x)
        posterior = self.distribution(z)
        z = posterior.sample()
        x_hat = self.decoder(z)
        
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = torch.sum(posterior.kl())
        return (L1 + L2) / batch_size