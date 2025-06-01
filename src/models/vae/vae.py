import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.registry import GLOBAL_REGISTRY

@GLOBAL_REGISTRY.register(category='vae', name='BaseVAE')
class BaseVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        z = self.encoder(x)
        from src.factories.model_factory import get_model
        posterior = get_model(category='distribution', name='DiagonalGaussianDistribution', parameters=z)
        z = posterior.sample()
        x_hat = self.decoder(z)
        
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = torch.sum(posterior.kl())
        return (L1 + L2) / batch_size, z, x_hat

    def sample(self, noise):
        return self.decoder(noise)