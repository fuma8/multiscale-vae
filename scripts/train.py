import yaml
import torch

from src.registry.registry import GLOBAL_REGISTRY
from src.factories.dataset_factory import get_dataloader

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config_path = './config/config.yaml'
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_args = config['dataset']['args']
    dataloader = get_dataloader(dataset_name=config['dataset']['name'], **dataset_args)
    

if __name__ == "__main__":
    main()