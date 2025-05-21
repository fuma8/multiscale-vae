from src.registry.registry import GLOBAL_REGISTRY
from src.models.encoder.small_encoder import SmallEncoder
from src.models.encoder.diffusers_encoder import DiffusersEncoder
from src.models.decoder.small_decoder import SmallDecoder
from src.models.decoder.diffusers_decoder import DiffusersDecoder
from src.models.distributions.diagonal_gaussian import DiagonalGaussianDistribution
from src.models.vae.vae import BaseVAE

def get_model(category, name, **model_args):
    return GLOBAL_REGISTRY.get_instance(category=category, name=name, **model_args)