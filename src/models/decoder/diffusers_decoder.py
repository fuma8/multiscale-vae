from diffusers.models.autoencoders.vae import Decoder

from src.registry.registry import GLOBAL_REGISTRY

@GLOBAL_REGISTRY.register(category='decoder', name='DiffusersDecoder')
class DiffusersDecoder(Decoder):
    pass  