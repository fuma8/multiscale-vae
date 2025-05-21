from diffusers.models.autoencoders.vae import Encoder

from src.registry.registry import GLOBAL_REGISTRY

@GLOBAL_REGISTRY.register(category='encoder', name='DiffusersEncoder')
class DiffusersEncoder(Encoder):
    pass  