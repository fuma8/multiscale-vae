import yaml
import torch

from src.registry.registry import GLOBAL_REGISTRY
from src.factories.dataset_factory import get_dataloader
from src.factories.model_factory import get_model

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config_path = './config/config.yaml'
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_args = config['dataset']['args']
    dataloader = get_dataloader(dataset_name=config['dataset']['name'], **dataset_args)
    
    encoder = get_model(category='encoder', name=config['model']['encoder'])
    decoder = get_model(category='decoder', name=config['model']['decoder'])

    vae = get_model(category='vae', name=config['model']['vae'], encoder=encoder, decoder=decoder)
    vae = vae.to(device)
    
    
if __name__ == "__main__":
    main()