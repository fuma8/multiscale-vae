# This file is adapted from:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py
#
# Original source is part of the diffusers library by Hugging Face.
# Licensed under the Apache License, Version 2.0.
# See https://www.apache.org/licenses/LICENSE-2.0 for license details.

from src.registry.registry import GLOBAL_REGISTRY

@GLOBAL_REGISTRY.register(catetgory='distribution', name='diagonal_gaussian_distribution')
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x